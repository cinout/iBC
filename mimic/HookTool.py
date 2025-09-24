class HookTool:
    def __init__(self):
        self.fea_out = None
        self.fea_map = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea_map = fea_out  # non-flatten one
        self.fea_out = fea_out.flatten(start_dim=1)  # [B, dim]


def get_feas_by_hook(model, layer_names=None):
    if layer_names is None:
        raise Exception("layer_names cannot be empty!")

    fea_hooks = {}
    for n, m in model.named_modules():
        if n in layer_names:
            cur_hook = HookTool()
            m.register_forward_hook(
                cur_hook.hook_fun
            )  # register_forward_hook runs after the forward pass of that layer, but before the next layer starts
            fea_hooks[n] = cur_hook
    return fea_hooks
