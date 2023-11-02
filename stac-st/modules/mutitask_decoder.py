"""
    Decoding methods for seq2seq autoregressive model that allows the use of:
    source and target language tokens. Useful for ASR+ST models.

Authors
 * Juan Zuluaga-Gomez 2023
"""
import speechbrain as sb
import torch
from speechbrain.decoders.ctc import CTCPrefixScorer
from speechbrain.decoders.seq2seq import S2SBeamSearcher


class S2SMultiTaskTransformerBeamSearch(S2SBeamSearcher):
    """This class implements the beam search decoding
    for a Multitask Transformer.
    This model contains inputs in the decoder similar to
    OpenAI's Whisper model: https://cdn.openai.com/papers/whisper.pdf.

    See also S2SBaseSearcher(), S2SBeamSearcher().

    This class is a mix of:
        - S2SWhisperBeamSearch() and S2STransformerBeamSearch()

    Arguments
    ---------
    modules : list with the followings one:
        model : torch.nn.Module
            A Transformer model. It should have a decode() method.
        seq_lin : torch.nn.Module (mandatory)
            A linear output layer for the model.
        ctc_lin : torch.nn.Module (optional)
            A linear output layer for CTC.
    source_lang : int
        The token to define source language.
    target_lang : int
        The token to define target language.
    bos_token : int
        The token to use for beginning of sentence.
    timestamp_token : int
        The token to use for timestamp.
    max_length : int
        The maximum decoding steps to perform.
        The Whisper model has a maximum length of 448.
    **kwargs
        Arguments to pass to S2SBeamSearcher
    """

    def __init__(
        self,
        modules,
        temperature=1.0,
        temperature_lm=1.0,
        source_lang=-100,  # token id for source_lang
        target_lang=-100,  # token id for tar
        **kwargs,
    ):
        super(S2SMultiTaskTransformerBeamSearch, self).__init__(**kwargs)

        # we load the transformer model, CTC layer and output layer
        self.model = modules[0]
        self.fc = modules[1]
        self.ctc_fc = modules[2]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.temperature = temperature
        self.temperature_lm = temperature_lm

        # input tokens to the decoder
        self.decoder_input_tokens = None
        self.source_lang = source_lang  # default source language is spanish
        self.target_lang = target_lang  # default target language is english

        self.bos_token = self.bos_index  # always this value

    def set_source_language(self, source_lang):
        """set the source language token to use for the decoder input."""
        self.source_lang = source_lang

    def set_target_language(self, target_lang):
        """set the target language token to use for the decoder input."""
        # this if we want to instantiate one object that does both tasks
        self.target_lang = target_lang

    def set_decoder_prefix_tokens(self, source_lang, target_lang):
        """decoder_input_tokens are the tokens used as input to the decoder.
        They are directly taken from the Tokenizer.

            decoder_input_tokens = [bos_token, source_lang, target_lang]
        """
        self.set_source_language(source_lang)
        self.set_target_language(target_lang)

        # bos will be appended during Beam Search to this set of task tokens
        self.decoder_input_tokens = [
            self.bos_token,
            self.source_lang,
            self.target_lang,
        ]

    def reset_mem(self, batch_size, device):
        """This method set the first tokens to be decoder_input_tokens during search."""
        return torch.tensor([self.decoder_input_tokens] * batch_size).to(device)

    def reset_lm_mem(self, batch_size, device):
        """Needed to reset the LM memory during beamsearch."""
        return None

    def permute_mem(self, memory, index):
        """Permutes the memory."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def permute_lm_mem(self, memory, index):
        """Permutes the memory of the language model."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""

        # we need to check if the bos_token was already appended,
        # IF ALL inp_tokens==bos_index, we skip this step
        if not torch.all(inp_tokens == self.bos_index):
            memory = _update_mem(inp_tokens, memory)
        pred, attn = self.model.decode(memory, enc_states)
        prob_dist = self.softmax(self.fc(pred) / self.temperature)
        return prob_dist[:, -1, :], memory, attn

    def lm_forward_step(self, inp_tokens, memory):
        """Performs a step in the implemented LM module."""
        memory = _update_mem(inp_tokens, memory)
        if not next(self.lm_modules.parameters()).is_cuda:
            self.lm_modules.to(inp_tokens.device)
        logits = self.lm_modules(memory)
        log_probs = self.softmax(logits / self.temperature_lm)
        return log_probs[:, -1, :], memory


def _update_mem(inp_tokens, memory):
    """This function is for updating the memory for transformer searches.
    it is called at each decoding step. When being called, it appends the
    predicted token of the previous step to existing memory.

    Arguments:
    -----------
    inp_tokens : tensor
        Predicted token of the previous decoding step.
    memory : tensor
        Contains all the predicted tokens.
    """
    if memory is None:
        return inp_tokens.unsqueeze(1)
    return torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)
