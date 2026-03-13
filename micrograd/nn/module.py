"""
Base Module class — the foundation for all neural network components.
Mimics PyTorch's nn.Module with parameter registration and recursive utilities.
"""


class Module:
    """
    Base class for all neural network modules.

    Subclasses should:
    1. Call super().__init__() in their __init__
    2. Implement the forward() method
    3. Assign Tensor parameters as attributes (they will be auto-registered)
    4. Assign submodules as attributes (they will be auto-registered)
    """

    def __init__(self):
        # Use object.__setattr__ to bypass our custom __setattr__ during init
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, 'training', True)

    def __setattr__(self, name, value):
        """
        Custom attribute setter that automatically registers:
        - Tensor with requires_grad=True → _parameters
        - Module subclasses → _modules
        - Everything else → normal attribute
        """
        from ..tensor import Tensor

        # Remove from existing registries if re-assigning
        params = object.__getattribute__(self, '_parameters')
        modules = object.__getattribute__(self, '_modules')

        if name in params:
            del params[name]
        if name in modules:
            del modules[name]

        if isinstance(value, Tensor) and value.requires_grad:
            params[name] = value
        elif isinstance(value, Module):
            modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """Look up parameters and modules when attribute not found normally."""
        # This is only called when normal attribute lookup fails
        try:
            params = object.__getattribute__(self, '_parameters')
            if name in params:
                return params[name]
        except AttributeError:
            pass
        try:
            modules = object.__getattribute__(self, '_modules')
            if name in modules:
                return modules[name]
        except AttributeError:
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def parameters(self):
        """
        Yield all parameters recursively (including those in submodules).
        Returns a generator of Tensor objects with requires_grad=True.
        """
        # Yield own parameters
        params = object.__getattribute__(self, '_parameters')
        for param in params.values():
            yield param

        # Recurse into submodules
        modules = object.__getattribute__(self, '_modules')
        for module in modules.values():
            yield from module.parameters()

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        """Set the module in training mode."""
        object.__setattr__(self, 'training', mode)
        modules = object.__getattribute__(self, '_modules')
        for m in modules.values():
            m.train(mode)
        return self

    def eval(self):
        """Set the module in evaluation mode (disables dropout, etc.)."""
        return self.train(False)

    def __call__(self, *args, **kwargs):
        """Make the module callable — delegates to forward()."""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Define the computation performed at every call.
        Subclasses must override this.
        """
        raise NotImplementedError(
            f"Module '{type(self).__name__}' must implement forward()"
        )

    def __repr__(self):
        """
        Returns a string representation similar to PyTorch's module repr.
        Example:
            Sequential(
              (0): Linear(in_features=2, out_features=8)
              (1): ReLU()
            )
        """
        modules = object.__getattribute__(self, '_modules')
        params = object.__getattribute__(self, '_parameters')

        if not modules and not params:
            return f"{type(self).__name__}()"

        lines = [f"{type(self).__name__}("]
        for key, module in modules.items():
            mod_str = repr(module)
            # Indent sub-module representation
            mod_str_indented = mod_str.replace('\n', '\n  ')
            lines.append(f"  ({key}): {mod_str_indented}")
        lines.append(")")
        return '\n'.join(lines)
