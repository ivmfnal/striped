# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.password = 'sha1:bf77ed5186d2:3ed1a075e13327f772c68b8a5d48e9fa6f6936d2'
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 8444
c.NotebookApp.certfile = "/home/ivm/.cert/server.crt"
c.NotebookApp.keyfile = "/home/ivm/.cert/server.key"
c.NotebookApp.notebook_dir = "./notebook"
c.NotebookApp.ssl_options = {"ssl_version":5}


