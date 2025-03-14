import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss


class Model(nn.Module):
    def __init__(self, encoder, encoder_momentum, args):
        super(Model, self).__init__()
        self.K = 8192  # queue size
        self.m = 0.999  # momentum coefficient
        self.T = 0.05  # temperature coefficient
        self.args = args
        self.encoder = encoder
        self.encoder_momentum = encoder_momentum
        for param, param_m in zip(self.encoder.parameters(), self.encoder_momentum.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        self.register_buffer("queue_code", torch.randn(768, self.K))
        self.register_buffer("queue_nl", torch.randn(768, self.K))
        self.queue_code = nn.functional.normalize(self.queue_code, dim=0)
        self.queue_nl = nn.functional.normalize(self.queue_nl, dim=0)
        self.register_buffer("queue_ptr_code", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_nl", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update encoder
        """
        for param, param_m in zip(self.encoder.parameters(), self.encoder_momentum.parameters()):
            param_m.data = param_m.data * self.m + param.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_nl(self, nl_vec):
        """
        NL Queue dequeue and enqueue
        """
        nl_size = nl_vec.shape[0]
        ptr = int(self.queue_ptr_nl)
        if nl_size == self.args.train_batch_size:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue_nl[:, ptr:ptr + nl_size] = nl_vec.T
            ptr = (ptr + nl_size) % self.K  # move pointer
            self.queue_ptr_nl[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_code(self, code_vec):
        """
        PL Queue dequeue and enqueue
        """
        code_size = code_vec.shape[0]
        ptr = int(self.queue_ptr_code)
        if code_size == self.args.train_batch_size:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue_code[:, ptr:ptr + code_size] = code_vec.T
            ptr = (ptr + code_size) % self.K  # move pointer
            self.queue_ptr_code[0] = ptr

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None and nl_inputs is not None:
            outputs_code = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            code_vec = (outputs_code*code_inputs.ne(1)[:, :, None]).sum(1)/code_inputs.ne(1).sum(-1)[:, None]
            code_vec = torch.nn.functional.normalize(code_vec, p=2, dim=1)
            outputs_nl = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            nl_vec = (outputs_nl*nl_inputs.ne(1)[:, :, None]).sum(1)/nl_inputs.ne(1).sum(-1)[:, None]
            nl_vec = torch.nn.functional.normalize(nl_vec, p=2, dim=1)

            with torch.no_grad():
                self._momentum_update_key_encoder()
                outputs_code_m = self.encoder_momentum(code_inputs, attention_mask=code_inputs.ne(1))[0]
                code_vec_m = (outputs_code_m*code_inputs.ne(1)[:, :, None]).sum(1)/code_inputs.ne(1).sum(-1)[:, None]
                code_vec_m = torch.nn.functional.normalize(code_vec_m, p=2, dim=1)
                outputs_nl_m = self.encoder_momentum(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
                nl_vec_m = (outputs_nl_m*nl_inputs.ne(1)[:, :, None]).sum(1)/nl_inputs.ne(1).sum(-1)[:, None]
                nl_vec_m = torch.nn.functional.normalize(nl_vec_m, p=2, dim=1)

            l_pos_code = torch.einsum('nc,nc->n', [code_vec, nl_vec_m]).unsqueeze(-1)
            l_pos_nl = torch.einsum('nc,nc->n', [nl_vec, code_vec_m]).unsqueeze(-1)
            l_neg_code = torch.einsum('nc,ck->nk', [code_vec, self.queue_nl.clone().detach()])
            l_neg_nl = torch.einsum('nc,ck->nk', [nl_vec, self.queue_code.clone().detach()])
            logits_code = torch.cat([l_pos_code, l_neg_code], dim=1)
            logits_nl = torch.cat([l_pos_nl, l_neg_nl], dim=1)
            logits_code /= self.T
            logits_nl /= self.T

            loss_fct = CrossEntropyLoss()
            loss_code = loss_fct(logits_code, torch.zeros(logits_code.size(0), dtype=torch.long).cuda())
            loss_nl = loss_fct(logits_nl, torch.zeros(logits_nl.size(0), dtype=torch.long).cuda())
            loss = loss_code + loss_nl

            self._dequeue_and_enqueue_code(code_vec_m)
            self._dequeue_and_enqueue_nl(nl_vec_m)

            return loss, code_vec, nl_vec

        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            code_vec = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(code_vec, p=2, dim=1)

        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            nl_vec = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(nl_vec, p=2, dim=1)
