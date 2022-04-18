from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel


class TaggerRewriteModel(nn.Module):
    def __init__(self, config,args):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained(args.model_path)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start=None,
        end=None,
        insert_pos=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits, insert_pos_logits= logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        insert_pos_logits= insert_pos_logits.squeeze(-1)

        outputs = (start_logits, end_logits, insert_pos_logits)
        if start is not None:
            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start)
            end_loss = loss_fct(end_logits, end)
            insert_loss = loss_fct(insert_pos_logits, insert_pos)
            total_loss = (start_loss + end_loss + insert_loss) / 3
            outputs = (total_loss,) + outputs
            return outputs
        else:
            return (None,) + outputs

