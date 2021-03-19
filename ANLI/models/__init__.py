__version__ = "3.0.2"

# from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, AutoConfig
# from .configuration_bart import BartConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
# from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
# from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
# from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
# from .configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
# from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
# from .configuration_encoder_decoder import EncoderDecoderConfig
# from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
# from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
# from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
# from .configuration_marian import MarianConfig
# from .configuration_mbart import MBartConfig
# from .configuration_mmbt import MMBTConfig
# from .configuration_mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileBertConfig
# from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
# from .configuration_pegasus import PegasusConfig
# from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
# from .configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
# from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
# from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig
# from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
# from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
# from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig

# Tokenizers
# from .tokenization_albert import AlbertTokenizer
from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
# from .tokenization_bart import BartTokenizer, BartTokenizerFast
from .tokenization_bert import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
# from .tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
# from .tokenization_camembert import CamembertTokenizer
# from .tokenization_ctrl import CTRLTokenizer
# from .tokenization_distilbert import DistilBertTokenizer, DistilBertTokenizerFast
# from .tokenization_dpr import (
#     DPRContextEncoderTokenizer,
#     DPRContextEncoderTokenizerFast,
#     DPRQuestionEncoderTokenizer,
#     DPRQuestionEncoderTokenizerFast,
#     DPRReaderTokenizer,
#     DPRReaderTokenizerFast,
# )
# from .tokenization_electra import ElectraTokenizer, ElectraTokenizerFast
# from .tokenization_flaubert import FlaubertTokenizer
# from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
# from .tokenization_longformer import LongformerTokenizer, LongformerTokenizerFast
# from .tokenization_mbart import MBartTokenizer
# from .tokenization_mobilebert import MobileBertTokenizer, MobileBertTokenizerFast
# from .tokenization_openai import OpenAIGPTTokenizer, OpenAIGPTTokenizerFast
# from .tokenization_pegasus import PegasusTokenizer
# from .tokenization_reformer import ReformerTokenizer
# from .tokenization_retribert import RetriBertTokenizer, RetriBertTokenizerFast
from .tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
# from .tokenization_t5 import T5Tokenizer
# from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer, TransfoXLTokenizerFast
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
     BatchEncoding,
     CharSpan,
     PreTrainedTokenizerBase,
     SpecialTokensMixin,
     TensorType,
     TokenSpan,
)
from .tokenization_utils_fast import PreTrainedTokenizerFast
# from .tokenization_xlm import XLMTokenizer
# from .tokenization_xlm_roberta import XLMRobertaTokenizer
# from .tokenization_xlnet import SPIECE_UNDERLINE, XLNetTokenizer

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_psutil_available,
    is_py3nvml_available,
    is_tf_available,
    is_torch_available,
    is_torch_tpu_available,
)

from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, apply_chunking_to_forward

from .modeling_auto import (
        AutoModel,
        AutoModelForPreTraining,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelWithLMHead,
        #AutoModelForCausalLM,
        #AutoModelForMaskedLM,
        #AutoModelForSeq2SeqLM,
        AutoModelForTokenClassification,
        AutoModelForMultipleChoice,
        MODEL_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        #MODEL_FOR_CAUSAL_LM_MAPPING,
       #MODEL_FOR_MASKED_LM_MAPPING,
        #MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    )

    # from .modeling_mobilebert import (
    #     MobileBertPreTrainedModel,
    #     MobileBertModel,
    #     MobileBertForPreTraining,
    #     MobileBertForSequenceClassification,
    #     MobileBertForQuestionAnswering,
    #     MobileBertForMaskedLM,
    #     MobileBertForNextSentencePrediction,
    #     MobileBertForMultipleChoice,
    #     MobileBertForTokenClassification,
    #     load_tf_weights_in_mobilebert,
    #     MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    #     MobileBertLayer,
    # )

from .bert import (
        BertPreTrainedModel,
        BertModel,
        BertForPreTraining,
        BertForMaskedLM,
        BertLMHeadModel,
        BertForNextSentencePrediction,
        BertForSequenceClassification,
        BertForMultipleChoice,
        BertForTokenClassification,
        BertForQuestionAnswering,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertLayer,
    )
    # from .modeling_openai import (
    #     OpenAIGPTPreTrainedModel,
    #     OpenAIGPTModel,
    #     OpenAIGPTLMHeadModel,
    #     OpenAIGPTDoubleHeadsModel,
    #     load_tf_weights_in_openai_gpt,
    #     OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_transfo_xl import (
    #     TransfoXLPreTrainedModel,
    #     TransfoXLModel,
    #     TransfoXLLMHeadModel,
    #     AdaptiveEmbedding,
    #     load_tf_weights_in_transfo_xl,
    #     TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_gpt2 import (
    #     GPT2PreTrainedModel,
    #     GPT2Model,
    #     GPT2LMHeadModel,
    #     GPT2DoubleHeadsModel,
    #     load_tf_weights_in_gpt2,
    #     GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_ctrl import CTRLPreTrainedModel, CTRLModel, CTRLLMHeadModel, CTRL_PRETRAINED_MODEL_ARCHIVE_LIST
    # from .modeling_xlnet import (
    #     XLNetPreTrainedModel,
    #     XLNetModel,
    #     XLNetLMHeadModel,
    #     XLNetForSequenceClassification,
    #     XLNetForTokenClassification,
    #     XLNetForMultipleChoice,
    #     XLNetForQuestionAnsweringSimple,
    #     XLNetForQuestionAnswering,
    #     load_tf_weights_in_xlnet,
    #     XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_xlm import (
    #     XLMPreTrainedModel,
    #     XLMModel,
    #     XLMWithLMHeadModel,
    #     XLMForSequenceClassification,
    #     XLMForTokenClassification,
    #     XLMForQuestionAnswering,
    #     XLMForQuestionAnsweringSimple,
    #     XLMForMultipleChoice,
    #     XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_pegasus import PegasusForConditionalGeneration
    # from .modeling_bart import (
    #     PretrainedBartModel,
    #     BartForSequenceClassification,
    #     BartModel,
    #     BartForConditionalGeneration,
    #     BartForQuestionAnswering,
    #     BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_mbart import MBartForConditionalGeneration
    # from .modeling_marian import MarianMTModel
    # from .tokenization_marian import MarianTokenizer
from .roberta import (
        RobertaForMaskedLM,
        #RobertaForCausalLM,
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaForMultipleChoice,
        RobertaForTokenClassification,
        RobertaForQuestionAnswering,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    # from .modeling_distilbert import (
    #     DistilBertPreTrainedModel,
    #     DistilBertForMaskedLM,
    #     DistilBertModel,
    #     DistilBertForMultipleChoice,
    #     DistilBertForSequenceClassification,
    #     DistilBertForQuestionAnswering,
    #     DistilBertForTokenClassification,
    #     DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_camembert import (
    #     CamembertForMaskedLM,
    #     CamembertModel,
    #     CamembertForSequenceClassification,
    #     CamembertForMultipleChoice,
    #     CamembertForTokenClassification,
    #     CamembertForQuestionAnswering,
    #     CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_encoder_decoder import EncoderDecoderModel
    # from .modeling_t5 import (
    #     T5PreTrainedModel,
    #     T5Model,
    #     T5ForConditionalGeneration,
    #     load_tf_weights_in_t5,
    #     T5_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_albert import (
    #     AlbertPreTrainedModel,
    #     AlbertModel,
    #     AlbertForPreTraining,
    #     AlbertForMaskedLM,
    #     AlbertForMultipleChoice,
    #     AlbertForSequenceClassification,
    #     AlbertForQuestionAnswering,
    #     AlbertForTokenClassification,
    #     load_tf_weights_in_albert,
    #     ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_xlm_roberta import (
    #     XLMRobertaForMaskedLM,
    #     XLMRobertaModel,
    #     XLMRobertaForMultipleChoice,
    #     XLMRobertaForSequenceClassification,
    #     XLMRobertaForTokenClassification,
    #     XLMRobertaForQuestionAnswering,
    #     XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )
    # from .modeling_mmbt import ModalEmbeddings, MMBTModel, MMBTForClassification

    # from .modeling_flaubert import (
    #     FlaubertModel,
    #     FlaubertWithLMHeadModel,
    #     FlaubertForSequenceClassification,
    #     FlaubertForTokenClassification,
    #     FlaubertForQuestionAnswering,
    #     FlaubertForQuestionAnsweringSimple,
    #     FlaubertForTokenClassification,
    #     FlaubertForMultipleChoice,
    #     FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )

    # from .modeling_electra import (
    #     ElectraForPreTraining,
    #     ElectraForMaskedLM,
    #     ElectraForTokenClassification,
    #     ElectraPreTrainedModel,
    #     ElectraForMultipleChoice,
    #     ElectraForSequenceClassification,
    #     ElectraForQuestionAnswering,
    #     ElectraModel,
    #     load_tf_weights_in_electra,
    #     ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )

    # from .modeling_reformer import (
    #     ReformerAttention,
    #     ReformerLayer,
    #     ReformerModel,
    #     ReformerForMaskedLM,
    #     ReformerModelWithLMHead,
    #     ReformerForSequenceClassification,
    #     ReformerForQuestionAnswering,
    #     REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )

    # from .modeling_longformer import (
    #     LongformerModel,
    #     LongformerForMaskedLM,
    #     LongformerForSequenceClassification,
    #     LongformerForMultipleChoice,
    #     LongformerForTokenClassification,
    #     LongformerForQuestionAnswering,
    #     LongformerSelfAttention,
    #     LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )

    # from .modeling_dpr import (
    #     DPRPretrainedContextEncoder,
    #     DPRPretrainedQuestionEncoder,
    #     DPRPretrainedReader,
    #     DPRContextEncoder,
    #     DPRQuestionEncoder,
    #     DPRReader,
    # )
    # from .modeling_retribert import (
    #     RetriBertPreTrainedModel,
    #     RetriBertModel,
    #     RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    # )

from .hf_argparser import HfArgumentParser

from .trainer_utils import EvalPrediction, set_seed

# Optimization
from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
)


