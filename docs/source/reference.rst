Striped Framework Reference
===========================

.. py:class:: Worker

    Worker is user-defined class, which specifies the code executed by the workers.  
    The user has to define the Worker class with the following members:

      * Object constructor
      * Columns - either class member or the Worker instance attribute - list of column names used for the analysis
      * frame() method, which will be called by the Framework for each data frame
      * end() method, which will be called after the last frame is processed by the Worker

    .. code-block:: python

        class Worker:
    
            Columns = [...]
        
            def __init__(self, user_params, bulk_data, job, db):
                # ...
        
            def frame(self, objects):
                # process frame
                return data
                
            def end(self):
                # called once after last frame
                return data

    The Worker obejct is created once per
    job per worker and it persists until the worker finishes running through its set of frames. That makes it possible to
    store and accumulate some data between processing the frames.

  .. py:method:: __init__(self, user_params, bulk_data, job, database)
  
      :param dictionary user_params: job parameters passed by the user to the Session.createJob() function as user_params argument
      :param dictionary bulk_data: bulk data dictionary passed to Session.createJob()
      :param object job: an object providing an interface to communicate back to the Job
      :param object database: the database object can be used to update data in the Striped database

  .. py:method:: frame(self, data)
  
      this method will be called once for each frame of data to be processed by the worker, synchronously.
  
      :param object data: frame accessor object providing access to the frame data
      :return: Either None or string "stop" or s data dictionary. The dictionary can have text keys and strings, integers, floating poing
               numbers or ndarrays as values. The dictionary can not be nested. If the worker's frame() method returns a dictionary, 
               this dictionary will be passed to the Accumulator's add() method.
      :raises StopIteration: alternative to returning "stop". If the method generates standard Python StopIteration exception, 
          the iteration through frames by this worker will stop

  .. py:method:: end(self) 

        this method will be called once after the worker finished processing its last frame.
        
    :return: either a dictionary with data to be sent to the Accumulator or None. The dictionary has the same restrictions as the
        dictionary returned by the frame() method.
        
