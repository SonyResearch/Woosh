import re
import torch
import string


def short_prompt(prompt: str, max_length=50):
    return f"{prompt[:max_length]}{'...' if len(prompt) > max_length else ''}"


class CLAPCaptionPostprocessTransform:
    def __init__(
        self,
        lowercase: bool = True,
        strip: bool = True,
        post_comma_space: bool = True,
        multiple_spaces: bool = True,
        remove_punctuation: bool = False,
        dropout_punctuation_prob: float = 0,
        caption_key: str = "captions",
    ):
        """
        remove_channels_captions: if True, removes the channel information from the captions
            Example channel keywords:
            x=x[:x.find("multi-mono")] # remove until the end
            "stereo,"
            "mono,"
            "multimono:"
            "stereo effect,"
            ", stereo"

        Args:
            lowercase (bool, optional): _description_. Defaults to True.
            strip (bool, optional): _description_. Defaults to True.
            post_comma_space (bool, optional): _description_. Defaults to True.
            multiple_spaces (bool, optional): _description_. Defaults to True.
            remove_punctuation (bool, optional): _description_. Defaults to False.
            remove_channels (bool, optional): _description_. Defaults to False.
            caption_key (str, optional): _description_. Defaults to "captions".
        """

        self.lowercase = lowercase
        self.strip = strip
        self.post_comma_space = post_comma_space
        self.multiple_spaces = multiple_spaces
        self.remove_punctuation = remove_punctuation
        self.remove_punctuation_func = str.maketrans("", "", string.punctuation)
        self.dropout_punctuation_prob = dropout_punctuation_prob

        self.caption_key = caption_key

    def __call__(self, batch):
        # expects captions to be a list of list of strings
        current_captions = batch[self.caption_key]
        return_lists = all(isinstance(c, list) for c in current_captions)
        return_strings = all(isinstance(c, str) or c is None for c in current_captions)
        assert return_lists or return_strings, (
            f"Expected captions to be a list of lists or a list of strings, got {current_captions}"
        )

        if return_strings:
            return_lists = False
            current_captions = [[c] for c in current_captions]

        processed = []
        for c in current_captions:
            c = [x or "" for x in c]
            if self.lowercase:
                c = [x.lower() for x in c]
            if self.strip:
                c = [x.strip() for x in c]

            if self.post_comma_space:
                # if there is no space after a comma, add one. also strip comma at the end if there is one
                c = [re.sub(r",([^ ])", r", \1", x).strip(",") for x in c]

            if self.multiple_spaces:
                # replace multiple spaces with a single space
                c = [re.sub(r"\s{2,}", " ", x) for x in c]

            if self.remove_punctuation:
                # removes all punctuation, including commas
                c = [x.translate(self.remove_punctuation_func) for x in c]
            elif self.dropout_punctuation_prob > 0:
                c = [
                    (
                        x.translate(self.remove_punctuation_func)
                        if torch.rand((1)).item() < self.dropout_punctuation_prob
                        else x
                    )
                    for x in c
                ]

            processed.append(c)
        if return_lists:
            batch[self.caption_key] = processed
        else:
            batch[self.caption_key] = [c[0] for c in processed]
        return batch
