from vega import VegaLite
import uuid
from IPython.display import publish_display_data, DisplayHandle
from ..common import Lockable, synchronized
from thread import get_ident

class UpdatableVegaLite(VegaLite):
    
    def __init__(self, *params, **args):
        VegaLite.__init__(self, *params, **args)
        self.ID = uuid.uuid4()
        self.InitNeeded = True
        
    def _ipython_display_(self):
        """Display the visualization in the Jupyter notebook."""
        if self.InitNeeded:
            html = self._generate_html(self.ID)
            #print "generated html:", html
            publish_display_data(
                {'text/html': html},
                metadata={}            
            )
        js = self._generate_js(self.ID)
        #print "generated javascript:", js
        publish_display_data(
            {'application/javascript': js},
            metadata={}                
        )
        self.InitNeeded = False
        
    def update_specs(self, specs):
        self.spec = specs

class IPythonDisplay(Lockable):
    
    def __init__(self, h):
        Lockable.__init__(self)
        self.H = h
        self.init()
        
    @synchronized
    def init(self):
        self.DisplayHandle = None
        self.VL = None
                   
    @synchronized      
    def display(self):
        self.DisplayHandle = DisplayHandle()
        self.VL = UpdatableVegaLite(self.H.vegalite())
        self.DisplayHandle.display(self.VL)
  
    @synchronized      
    def update(self):
        if self.DisplayHandle is None or self.VL is None:
            self.display()
        else:
            specs = self.H.vegalite()
            self.VL.update_specs(specs)
            #print "updaded specs"
            self.DisplayHandle.update(self.VL)
