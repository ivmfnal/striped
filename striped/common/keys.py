def stripe_key(dataset_name, column_name, rgid):
    return "%s:%s:%d.bin" % (dataset_name, column_name, rgid)

def rginfo_key(dataset_name, rgid):
    return "%s:@@rginfo:%d.json" % (dataset_name, rgid)

StripeHeaderFormatVersion = "1.0"

def stripe_header(array):
    return "#__header:version=%s;dtype=%s#" % (StripeHeaderFormatVersion, array.dtype.str)
    
def format_array(array):
    return bytes(stripe_header(array)) + bytes(array.data)
