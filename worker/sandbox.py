import sys, __builtin__, importlib

safe_modules = ["numpy","math","random","scipy","cmath","decimal","fractions","time",
    "numbers", "itertools", "operator",
	"numba","zlib","pickle","awkward","uproot_methods", "cloudpickle", "uproot", "fnal_column_analysis_tools"
]

safe_modules = {m:importlib.import_module(m) for m in safe_modules}

saved_import = __builtin__.__import__

def safe_import(name, *params):
    safe = (
	not name
	or name in safe_modules
	or name.startswith('.')
	or name.split(".")[0] in safe_modules
    )
    if not safe:
        raise ImportError("Can not import module '%s', params=%s" % (name, params))

    if not name or '.' in name:
        return saved_import(name, *params)
    else:
        return safe_modules[name]

builtin_remove = ["open","file","execfile"]
saved_builtins = {n:getattr(__builtin__, n) for n in builtin_remove}
    
def sandbox_call(func, *params, **args):
    try:
        __builtin__.__import__ = safe_import
        
        for n in builtin_remove:
            delattr(__builtin__, n)

        return func(*params, **args)
    except:
        raise
    finally:
        __builtin__.__import__ = saved_import
        for n, s in saved_builtins.items():
            setattr(__builtin__, n, s)

def sandbox_import_module(module_name, names):
    return sandbox_call(saved_import, module_name, {}, {}, names)
