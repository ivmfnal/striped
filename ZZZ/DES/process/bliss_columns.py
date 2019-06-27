import fitsio, sys, json

attrs = {
    "HPIX":{
            "dtype":    "<i8",
            "shape":    [],
            "subtype":  None,
            "source":   None
        },
    "EXPNUM":    {
        "dtype":    "<i8",
        "shape":    [],
        "subtype":  None,
        "source":   "EXPNUM"
    },
    "CCDNUM":    {
        "dtype":    "<i8",
        "shape":    [],
        "subtype":  None,
        "source":   "CCDNUM"
    },        
    "OBJECT_ID": {
        "dtype": "<i8", 
        "shape": [], 
        "source": null, 
        "subtype": null
    }               
}

data = fitsio.read(sys.argv[1], ext=2)

for fn, ft in data.dtype.fields.items():
    ft = ft[0].str
    if ft.startswith("|V"):
        width = data[fn].shape[-1]
        ft = data[fn].dtype.str.replace('>','<')
        attrs[fn] = {
            "dtype":  ft,
            "shape":  [width],
            "source": fn,
            "subtype":  None
        }
    else:
        ft = data[fn].dtype.str.replace('>','<')
        attrs[fn] = {
            "dtype":  ft,
            "shape":  [],
            "source": fn,
            "subtype":  None
        }

schema = {
    "version": "1.0",
    "attributes":   {
        "OBJECT_ID":    {
            "dtype":    "<i8",
            "shape":    [],
            "subtype":  None,
            "source":   None
        },
        "HPIX":    {
            "dtype":    "<i8",
            "shape":    [],
            "subtype":  None,
            "source":   None
        },
        "DELTAWIN_J2000":    {
            "dtype":    "<f8",
            "shape":    [],
            "subtype":  None,
            "source":   None
        },
        "ALPHAWIN_J2000":    {
            "dtype":    "<f8",
            "shape":    [],
            "subtype":  None,
            "source":   None
        },
        "BAD":  {
            "dtype":    "<i8",
            "shape":    [],
            "subtype":  None,
            "source":   None
        }
    },
    "branches":{
        "Observation":attrs
    },
    "tree_top":""
}
    
json.dump(schema, open(sys.argv[2], "w"), indent=4, sort_keys=True)
for fn, fd in attrs.items():
    print fn, fd["dtype"], fd["shape"]
