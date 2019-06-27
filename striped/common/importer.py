import sys, time, os

def import_from_text(module_text, names, module_name=None, tmp="/tmp"):
        saved_path = sys.path[:]
        imported = None
        try:
            if not tmp in sys.path:
                sys.path.insert(0, tmp)
            module_name = module_name or "m_%s_%d" % (os.getpid(), int(time.time()*1000.0))    
            assert not "/" in module_name 
            module_file = "%s/%s.py" % (tmp, module_name)
            open(module_file, "w").write(module_text)
            imported = __import__(module_name, {}, {}, names)
            try:
                os.unlink(module_file)
                os.unlink(module_file+"c")
            except:
                pass
        finally:
            sys.path = saved_path
            del sys.modules[module_name]
        return imported
