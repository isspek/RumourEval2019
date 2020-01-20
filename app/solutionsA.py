from ensemble_framework import Ensemble_Framework
from base_framework import Base_Framework
from base_framework_seq import Base_Framework_SEQ
from frameworks import BERT_Framework
from frameworks import BERT_Framework_for_veracity
from bert_framework_with_f import BERT_Framework_with_f
from bert_introspection_framework import BERT_Introspection_Framework
from self_att_with_bert_tokenizing import SelfAtt_BertTokenizing_Framework
from text_features_framework import Text_Feature_Framework
from text_framework_branch import Text_Framework
from text_framework_seq import Text_Framework_Seq
from baseline import Baseline
from models import Baseline_LSTM
from models import BertModelForStanceClassificationWFeatures
from models import SelfAttandBsline
from models import SelfAttWithBertTokenizing
from models import SelAttTextOnly, SelAttTextOnlyWithoutPrepInput
from models import BertModelForStanceClassification
from text_BERT_with_veracity import BertModelForVeracityClassification

__author__ = "Martin Fajčík"


class SolutionA:
    def __init__(self, config):
        self.config = config

    def create_model(self):
        """
        Create and validate model
        """
        modelf = None
        fworkf = Base_Framework
        if self.config["active_model"] == "baseline_LSTM":
            # 204 804 params
            modelf = Baseline_LSTM
        elif self.config["active_model"] == "baseline":
            modelf = Baseline
        elif self.config["active_model"] == "selfatt_textonly":
            modelf = SelAttTextOnly
            fworkf = Text_Framework
        elif self.config["active_model"] == "selfatt_textonly_seq":
            modelf = SelAttTextOnlyWithoutPrepInput
            fworkf = Text_Framework_Seq
        elif self.config["active_model"] == "selfatt_text_and_baseline":
            modelf = SelfAttandBsline
            fworkf = Text_Feature_Framework
        elif self.config["active_model"] == "BERT_textonly":
            modelf = BertModelForStanceClassification
            # fworkf = BERT_Framework_Hyperparamopt
            fworkf = BERT_Framework
        elif self.config["active_model"] == "features_seq":
            modelf = Baseline
            fworkf = Base_Framework_SEQ
        elif self.config["active_model"] == "BERT_withf":
            modelf = BertModelForStanceClassificationWFeatures
            fworkf = BERT_Framework_with_f

        # In fact, this was an useless experiment, since we have only ~300 source post classified for veracity
        elif self.config["active_model"] == "BERT_veracity":
            modelf = BertModelForVeracityClassification
            fworkf = BERT_Framework_for_veracity

        elif self.config["active_model"] == "self_att_with_bert_tokenizer":
            modelf = SelfAttWithBertTokenizing
            fworkf = SelfAtt_BertTokenizing_Framework

        elif self.config["active_model"] == "ensemble":
            modelf = BertModelForStanceClassification
            fworkf = Ensemble_Framework
        elif self.config["active_model"] == "BERT_introspection":
            modelf = BertModelForStanceClassification
            fworkf = BERT_Introspection_Framework
        modelframework = fworkf(self.config["models"][self.config["active_model"]])
        modelframework.fit(modelf)

    def submit_model(self, model):
        """
        Load model and run submission
        """
        pass
