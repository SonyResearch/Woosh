import os

import av
import numpy as np
import torch
import torchaudio


def remux_video(
    output_path,
    video_path=None,
    audio_input=None,
    sample_rate=48000,
    audio_start=0,
    video_chunk=None,
    duration_seconds=5,
):
    """
    Remux a video with new FLAC audio using PyAV.

    Args:
        output_path (str): Path to save the remuxed video.
        video_path (str): Path to the original video file (used if video_chunk is None).
        audio_input (str, np.ndarray, torch.Tensor): Path to the audio file, or audio array/tensor.
        sample_rate (int): Target sample rate for the audio.
        audio_start (int): Start time in seconds used to slice the video.
        video_chunk (np.ndarray, torch.Tensor): Pre-extracted video frames.
        duration_seconds (int): Duration of the video chunk to extract if video_chunk is None.

    Returns:
        torch.Tensor: The video chunk used for remuxing.
    """
    print(f"Remuxing to: {output_path}")
    frame_rate = 24

    # Process Audio Input

    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file missing: {audio_input}")

        # Load audio using torchaudio
        waveform, sr = torchaudio.load(audio_input)

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=sample_rate
            )
            waveform = resampler(waveform)

        audio_np = waveform.transpose(0, 1).numpy()

    elif isinstance(audio_input, (np.ndarray, torch.Tensor)):
        audio_np = (
            audio_input.detach().cpu().numpy()
            if isinstance(audio_input, torch.Tensor)
            else audio_input
        )
        if audio_np.ndim == 1:
            audio_np = np.expand_dims(audio_np, axis=-1)
        elif audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
            audio_np = audio_np.T
    else:
        raise ValueError(
            "audio_input must be a file path, numpy.ndarray, or torch.Tensor."
        )

    audio_np = audio_np.astype(np.float32)
    num_channels = audio_np.shape[1]

    # Process Video Input

    if video_chunk is None:
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file missing: {video_path}")

        v_container = av.open(video_path)
        v_stream = next(s for s in v_container.streams if s.type == "video")
        v_stream.codec_context.options = {"threads": "1"}
        v_stream.thread_type = "FRAME"

        start_time_sec = audio_start
        end_time_sec = start_time_sec + duration_seconds

        v_container.seek(
            int(start_time_sec * av.time_base), any_frame=False, backward=True
        )

        extracted_frames = []
        for packet in v_container.demux(v_stream):
            done_extracting = False
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * frame.time_base)

                if frame_time < start_time_sec:
                    continue
                if frame_time > end_time_sec:
                    done_extracting = True
                    break

                # frame = frame.reformat(width=640, height=480, format="rgb24")
                extracted_frames.append(frame.to_ndarray(format="rgb24"))

            if done_extracting:
                break
        v_container.close()

        video_chunk_np = (
            np.stack(extracted_frames)
            if extracted_frames
            else np.zeros((0, 480, 640, 3), dtype=np.uint8)
        )
    else:
        video_chunk_np = (
            video_chunk.detach().cpu().numpy()
            if isinstance(video_chunk, torch.Tensor)
            else video_chunk
        )

    if video_chunk_np.ndim == 4 and video_chunk_np.shape[1] in [1, 3]:
        video_chunk_np = np.transpose(video_chunk_np, (0, 2, 3, 1))

    if len(video_chunk_np) == 0:
        raise ValueError("Video chunk is empty.")

    t, h, w, c = video_chunk_np.shape

    # Muxing to Output

    writer = av.open(output_path, "w")

    # Setup Video Stream
    out_video = writer.add_stream("libx264", rate=frame_rate)
    out_video.width = w
    out_video.height = h
    out_video.pix_fmt = "yuv420p"

    # Setup Audio Stream (FLAC)
    out_audio = writer.add_stream("flac", rate=sample_rate)
    layout = "stereo" if num_channels == 2 else "mono"
    out_audio.layout = layout

    # Write Video
    for i in range(t):
        frame_arr = video_chunk_np[i]
        if frame_arr.dtype != np.uint8:
            frame_arr = (
                (frame_arr * 255).astype(np.uint8)
                if frame_arr.max() <= 1.0
                else frame_arr.astype(np.uint8)
            )

        v_frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        v_frame.pts = i
        for packet in out_video.encode(v_frame):
            writer.mux(packet)

    for packet in out_video.encode():
        writer.mux(packet)

    # Write Audio
    audio_np_int16 = (audio_np * 32767.0).clip(-32768, 32767).astype(np.int16)

    # Transpose to (Channels, Time) for planar encoding
    audio_np_t = audio_np_int16.T

    # FLAC frame sizes can vary; 4096 is a safe default if PyAV doesn't set one
    frame_size = out_audio.codec_context.frame_size or 4096
    num_samples = audio_np_t.shape[1]

    for offset in range(0, num_samples, frame_size):
        chunk = audio_np_t[:, offset : offset + frame_size]
        chunk = np.ascontiguousarray(chunk)

        # Use 's16p' (16-bit signed integer, planar) which FLAC expects
        a_frame = av.AudioFrame.from_ndarray(chunk, format="s16p", layout=layout)
        a_frame.sample_rate = sample_rate
        a_frame.pts = offset

        for packet in out_audio.encode(a_frame):
            writer.mux(packet)

    for packet in out_audio.encode():
        writer.mux(packet)

    writer.close()


def extract_video_frames(input_path, start_time=0, end_time=None):
    """Extract video frames from a video file between specified start and end times.

    Args:
        input_path (_type_): _path to the input video file.
        start_time (_type_): _start time in seconds for video extraction.
        end_time (_type_): _end time in seconds for video extraction.

    Returns:
        _type_: _tuple containing  video frames, video rate, and pts array.
    """
    container = av.open(input_path)

    video_frames = []

    # Assume first video and audio streams
    video_stream = next(s for s in container.streams if s.type == "video")

    # running on parallel we want single thread (2 threads per file)
    video_stream.codec_context.options = {"threads": "1"}
    video_stream.thread_type = "FRAME"  # type: ignore

    video_rate = video_stream.average_rate
    # Seek to just before the start time (in microseconds)
    container.seek(int(start_time * av.time_base), any_frame=False, backward=True)
    pts_arr = []
    for packet in container.demux((video_stream)):
        done_extracting = False
        for frame in packet.decode():
            # Convert pts to seconds
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * frame.time_base)
            if frame_time < start_time:
                # print(
                #     f"Skipping frame at {frame_time} seconds, before start time {start_time} seconds."
                # )
                continue
            if end_time is not None and frame_time > end_time:
                # print(
                #     f"Reached {frame_time} end time: {end_time} seconds, stopping extraction."
                # )
                done_extracting = True
                break

            if packet.stream.type == "video":
                # Resize and convert to numpy (HWC RGB)
                frame = frame.reformat(width=224, height=224, format="rgb24")
                video_array = frame.to_ndarray(format="rgb24")
                video_frames.append(video_array)
                pts_arr.append(frame_time)

        if done_extracting:
            break

    container.close()

    # Concatenate all audio frames into a single numpy array

    video_frames = (
        np.stack(video_frames) if video_frames else np.zeros((0, 3, 224, 224))
    )
    pts_arr = np.array(pts_arr)
    return torch.from_numpy(video_frames), int(video_rate), torch.from_numpy(pts_arr)
