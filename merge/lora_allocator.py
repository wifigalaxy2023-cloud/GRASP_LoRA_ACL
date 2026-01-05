import torch
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class LoRAAllocatorInputParameterInit():
    def __init__(self, model, delete_percent, ipt_threshold=0):
        dist.barrier()
        self.delete_percent = delete_percent
        self.ipt = {}
        self.ipt_dict = {}
        self.save_initial_lora(model)
        dist.barrier()

    def save_initial_lora(self, model):
        initial_lora_dict = {}
        with torch.no_grad():
            for layer_idx, decoder_layer in enumerate(model.model.layers):
                layer_initial_lora_dict = {}
                for n,p in decoder_layer.named_parameters():
                    if "lora_" in n:
                        layer_initial_lora_dict[n] = p.clone().detach().to("cpu")
                initial_lora_dict[layer_idx] = layer_initial_lora_dict
            self.initial_lora_dict = initial_lora_dict

    def calc_layer_ipt(self, decoder_layer, hidden_states_list, layernorm=None):
        if layernorm is None:
            layernorm = LlamaRMSNorm(4096, eps=1e-05).to(hidden_states_list[0].device)
        hidden_states_list = [layernorm(hidden_states) for hidden_states in hidden_states_list]
        concatenated_hidden_states = torch.cat(hidden_states_list, dim=1).squeeze(0)
        norm_hidden_states = concatenated_hidden_states.norm(p=2, dim=0)

        lora_A_params = {n.replace("_A", ""):p for (n,p) in decoder_layer.named_parameters() if "lora_A_weights" in n}
        lora_B_params = {n.replace("_B", ""):p for (n,p) in decoder_layer.named_parameters() if "lora_B_weights" in n}

        parameterwise_ipt_dict = {}
        for n in lora_A_params.keys():
            if lora_A_params[n].numel() == 0 or lora_B_params[n].numel() == 0:
                parameterwise_ipt_dict[f"{n}_A"] = None
                parameterwise_ipt_dict[f"{n}_B"] = None
            else:
                lora_A_param = lora_A_params[n].reshape(8, -1).detach().clone()
                lora_B_param = lora_B_params[n].reshape(-1, 8).detach().clone()
                lora_A_param.requires_grad = True
                lora_B_param.requires_grad = True

                with torch.enable_grad():
                    lora_param = torch.matmul(lora_B_param, lora_A_param)
                    lora_param_ipt = lora_param.abs() * norm_hidden_states

                    importance_sum = lora_param_ipt.sum()

                    importance_sum.backward(retain_graph=True)
                    grad_lora_A_param = lora_A_param.grad.clone()

                    lora_A_param.grad.zero_()
                    importance_sum.backward()
                    grad_lora_B_param = lora_B_param.grad.clone()

                parameterwise_ipt_dict[f"{n}_A"] = grad_lora_A_param.abs()
                parameterwise_ipt_dict[f"{n}_B"] = grad_lora_B_param.abs()

        return parameterwise_ipt_dict

    def delete_param(self, decoder_layer, layer_idx):
        with torch.no_grad():
            for n,p in decoder_layer.named_parameters():
                lora_param_name = f"model.layers.{layer_idx}.{n}"
                if "_A" in n:
                    lora_param_name = lora_param_name.replace("_A", "") + "_A"
                elif "_B" in lora_param_name:
                    lora_param_name = lora_param_name.replace("_B", "") + "_B"

                if lora_param_name in self.ipt_dict:
                    flattened_importances = self.ipt_dict[lora_param_name].flatten()

                    num_params = flattened_importances.numel()
                    num_replace = max(1,int(num_params * self.delete_percent/100))

                    low_values, low_indices = torch.topk(flattened_importances, num_replace, largest=False)

                    low_indices_unraveled = torch.unravel_index(low_indices, p.shape)
                    p[low_indices_unraveled] = self.initial_lora_dict[layer_idx][n].clone().detach().to(p.device)[low_indices_unraveled]

    def update_allocation(self, model, train_dataset):
        dist.barrier()
        with torch.no_grad():
            hidden_states_list = []
            for input_ids in train_dataset["input_ids"]:
                hidden_states = model(torch.tensor([input_ids]).to(model.device), output_hidden_states=True).hidden_states

                hidden_states_list.append([hs.cpu() for hs in hidden_states])
                del hidden_states
                torch.cuda.empty_cache()

            layernorm = LlamaRMSNorm(4096, eps=1e-05).to(model.device)

            for layer_idx, decoder_layer in enumerate(model.model.layers):
                ipt_is_not_inf = True
                hidden_states_for_layer = [hidden_states[layer_idx].to(model.device) for hidden_states in hidden_states_list]

                layer_ipt_dict = self.calc_layer_ipt(decoder_layer, hidden_states_for_layer, layernorm)

                for k,v in layer_ipt_dict.items():
                    if v == None:
                        ipt_is_not_inf = False
                    else:
                        self.ipt_dict[f"model.layers.{layer_idx}.{k}"] = v

                if ipt_is_not_inf:
                    self.delete_param(decoder_layer, layer_idx)
                dist.barrier()

        dist.barrier()


class LoRAAllocatorInputParameterZero(LoRAAllocatorInputParameterInit):
    def delete_param(self, decoder_layer, layer_idx):
        with torch.no_grad():
            for n,p in decoder_layer.named_parameters():
                lora_param_name = f"model.layers.{layer_idx}.{n}"
                if "_A" in n:
                    lora_param_name = lora_param_name.replace("_A", "") + "_A"
                elif "_B" in lora_param_name:
                    lora_param_name = lora_param_name.replace("_B", "") + "_B"

                if lora_param_name in self.ipt_dict:
                    flattened_importances = self.ipt_dict[lora_param_name].flatten()

                    num_params = flattened_importances.numel()
                    num_replace = max(1,int(num_params * self.delete_percent/100))

                    low_values, low_indices = torch.topk(flattened_importances, num_replace, largest=False)

                    low_indices_unraveled = torch.unravel_index(low_indices, p.shape)
                    p[low_indices_unraveled] = 0


class LoRAAllocatorInputModuleInit():
    def __init__(self, model, ipt_threshold, delete_percent=0):
        dist.barrier()
        self.ipt_threshold = ipt_threshold
        self.ipt = {}
        self.ipt_dict = {}
        self.save_initial_lora(model)
        dist.barrier()

    def save_initial_lora(self, model):
        initial_lora_dict = {}
        with torch.no_grad():
            for layer_idx, decoder_layer in enumerate(model.model.layers):
                layer_initial_lora_dict = {}
                for n,p in decoder_layer.named_parameters():
                    if "lora_" in n:
                        layer_initial_lora_dict[n] = p.clone().detach().to("cpu")
                initial_lora_dict[layer_idx] = layer_initial_lora_dict
            self.initial_lora_dict = initial_lora_dict

    def calc_threshold(self):
        return self.ipt_threshold

    def calc_layer_ipt(self, decoder_layer, hidden_states_list, layernorm=None):
        if layernorm is None:
            layernorm = LlamaRMSNorm(4096, eps=1e-05).to(hidden_states_list[0].device)
        hidden_states_list = [layernorm(hidden_states) for hidden_states in hidden_states_list]
        concatenated_hidden_states = torch.cat(hidden_states_list, dim=1).squeeze(0)
        norm_hidden_states = concatenated_hidden_states.norm(p=2, dim=0)

        lora_A_params = {n.replace("_A", ""):p for (n,p) in decoder_layer.named_parameters() if "lora_A_weights" in n}
        lora_B_params = {n.replace("_B", ""):p for (n,p) in decoder_layer.named_parameters() if "lora_B_weights" in n}

        parameterwise_ipt_dict = {}
        for n in lora_A_params.keys():
            if lora_A_params[n].numel() == 0 or lora_B_params[n].numel() == 0:
                parameterwise_ipt_dict[n] = float("inf")
            else:
                lora_param = torch.matmul(lora_B_params[n].reshape(-1, 8), lora_A_params[n].reshape(8, -1))
                parameterwise_ipt = lora_param.abs() * norm_hidden_states
                parameterwise_ipt_dict[n] = torch.mean(parameterwise_ipt)

        return parameterwise_ipt_dict

    def delete_param(self, decoder_layer, layer_idx):
        ipt_threshold = self.calc_threshold()

        with torch.no_grad():
            for n,p in decoder_layer.named_parameters():
                lora_param_name = n.replace("_A", "").replace("_B", "")
                lora_param_name = f"model.layers.{layer_idx}.{lora_param_name}"
                if (lora_param_name in self.ipt_dict) and (self.ipt_dict[lora_param_name] <= ipt_threshold):
                    p.data = self.initial_lora_dict[layer_idx][n].clone().detach().to(p.device)
                    print(f"- Delete {n} (layer{layer_idx})")

    def update_allocation(self, model, train_dataset):
        dist.barrier()
        with torch.no_grad():
            hidden_states_list = [
                model(torch.tensor([input_ids]).to(model.device), output_hidden_states=True).hidden_states for input_ids in train_dataset["input_ids"]
            ]
            layernorm = LlamaRMSNorm(4096, eps=1e-05).to(model.device)

            for layer_idx, decoder_layer in enumerate(model.model.layers):
                ipt_is_not_inf = True
                layer_ipt_dict = self.calc_layer_ipt(decoder_layer, [hidden_states[layer_idx] for hidden_states in hidden_states_list], layernorm)
                for k,v in layer_ipt_dict.items():
                    if v == float("inf"):
                        ipt_is_not_inf = False
                    else:
                        self.ipt_dict[f"model.layers.{layer_idx}.{k}"] = v

                if ipt_is_not_inf:
                    self.delete_param(decoder_layer,  layer_idx)
                dist.barrier()


class LoRAAllocatorInputModuleZero(LoRAAllocatorInputModuleInit):
    def delete_param(self, decoder_layer, layer_idx):
        ipt_threshold = self.calc_threshold()

        with torch.no_grad():
            for n,p in decoder_layer.named_parameters():
                lora_param_name = n.replace("_A", "").replace("_B", "")
                lora_param_name = f"model.layers.{layer_idx}.{lora_param_name}"
                if (lora_param_name in self.ipt_dict) and (self.ipt_dict[lora_param_name] <= ipt_threshold):
                    p.data.zero_()
                    print(f"- Delete {n} (layer{layer_idx})")


class LoRAAllocatorGradParameterInit():
    def __init__(self, model, delete_percent, beta1=0.85, beta2=0.85, ipt_threshold=0):
        dist.barrier()
        self.delete_percent = delete_percent
        self.beta1 = beta1
        self.beta2 = beta2
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.ipt_dict = {}
        self.save_initial_lora(model)
        dist.barrier()

    def save_initial_lora(self, model):
        initial_lora_dict = {}
        for n,p in model.named_parameters():
            if "lora_" in n:
                initial_lora_dict[n] = p.clone().detach().to("cpu")
        self.initial_lora_dict = initial_lora_dict

    def calc_parameterwise_ipt(self, model):
        for n,p in model.named_parameters():
            if "lora_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    if p.grad is not None:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    else:
                        self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                        (1-self.beta1) * self.ipt[n]
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                        (1-self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    self.ipt_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def delete_lora(self, model):
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in self.ipt_dict:
                    flattened_importances = self.ipt_dict[n].flatten()

                    num_params = flattened_importances.numel()
                    num_replace = max(1,int(num_params * self.delete_percent/100))

                    if num_params == 0:
                        continue

                    low_values, low_indices = torch.topk(flattened_importances, num_replace, largest=False)

                    low_indices_unraveled = torch.unravel_index(low_indices, p.shape)
                    p[low_indices_unraveled] = self.initial_lora_dict[n].clone().detach().to(p.device)[low_indices_unraveled]

    def update_allocation(self, model, train_dataset=None):
        dist.barrier()
        self.calc_parameterwise_ipt(model)
        self.delete_lora(model)
        dist.barrier()


class LoRAAllocatorGradParameterZero(LoRAAllocatorGradParameterInit):
    def delete_lora(self, model):
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in self.ipt_dict:
                    flattened_importances = self.ipt_dict[n].flatten()

                    num_params = flattened_importances.numel()
                    num_replace = max(1,int(num_params * self.delete_percent/100))

                    if num_params == 0:
                        continue

                    low_values, low_indices = torch.topk(flattened_importances, num_replace, largest=False)

                    low_indices_unraveled = torch.unravel_index(low_indices, p.shape)
                    p[low_indices_unraveled] = 0


class LoRAAllocatorGradModuleInit():
    def __init__(self, model, ipt_threshold, beta1=0.85, beta2=0.85, delete_percent=0):
        dist.barrier()
        self.ipt_threshold = ipt_threshold
        self.beta1 = beta1
        self.beta2 = beta2
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.ipt_dict = {}
        self.save_initial_lora(model)
        dist.barrier()

    def save_initial_lora(self, model):
        initial_lora_dict = {}
        for n,p in model.named_parameters():
            if "lora_" in n:
                initial_lora_dict[n] = p.clone().detach().to("cpu")
        self.initial_lora_dict = initial_lora_dict

    def calc_parameterwise_ipt(self, model):
        for n,p in model.named_parameters():
            if "lora_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    if p.grad is not None:
                        self.ipt[n] = (p * p.grad).abs().detach()
                    else:
                        self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                        (1-self.beta1) * self.ipt[n]
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                        (1-self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

    def calc_parameterwise_ipt(self, model):
        ipt_dict = {}
        for n,p in model.named_parameters():
            if "lora_" in n:
                ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
                avg_ipt = torch.mean(ipt_score)
                name_mat = n.replace("lora_A", "%s").replace("lora_B", "%s")
                if name_mat not in ipt_dict:
                    ipt_dict[name_mat] = [avg_ipt]
                else:
                    ipt_dict[name_mat].append(avg_ipt)

        for name_mat, ipt_list in ipt_dict.items():
            sum_ipt = torch.sum(torch.tensor(ipt_list))
            ipt_dict[name_mat] = sum_ipt

        return ipt_dict

    def calc_threshold(self):
        return self.ipt_threshold

    def delete_lora(self, model):
        ipt_dict = self.calc_parameterwise_ipt(model)
        ipt_threshold = self.calc_threshold()

        with torch.no_grad():
            for n,p in model.named_parameters():
                name_mat = n.replace("lora_A", "%s").replace("lora_B", "%s")
                if (name_mat in ipt_dict) and (ipt_dict[name_mat] <= ipt_threshold):
                    p.data = self.initial_lora_dict[n].clone().detach().to(p.device)

    def update_allocation(self, model, train_dataset=None):
        dist.barrier()
        self.calc_parameterwise_ipt(model)
        self.delete_lora(model)
        dist.barrier()


class LoRAAllocatorGradModuleZero(LoRAAllocatorGradModuleInit):
    def delete_lora(self, model):
        ipt_dict = self.calc_parameterwise_ipt(model)
        ipt_threshold = self.calc_threshold()

        with torch.no_grad():
            for n,p in model.named_parameters():
                name_mat = n.replace("lora_A", "%s").replace("lora_B", "%s")
                if (name_mat in ipt_dict) and (ipt_dict[name_mat] <= ipt_threshold):
                    p.data.zero_()


LORAALLOCATORDICT = {
    "input_parameter_init": LoRAAllocatorInputParameterInit,
    "input_parameter_zero": LoRAAllocatorInputParameterZero,
    "input_module_init": LoRAAllocatorInputModuleInit,
    "input_module_zero": LoRAAllocatorInputModuleZero,
    "grad_parameter_init": LoRAAllocatorGradParameterInit,
    "grad_parameter_zero": LoRAAllocatorGradParameterZero,
    "grad_module_init": LoRAAllocatorGradModuleInit,
    "grad_module_zero": LoRAAllocatorGradModuleZero,
}
