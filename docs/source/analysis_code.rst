User Analysis Code
==================

User analysis code consists of 3 parts - *Job*, *Worker* and *Accumulator*. The framework runs one Worker per
computing core, one Accumulator per worker node and one Job object per user job. Workers process data from their portions of the
dataset, running through lists of data frames. Each Worker can send some data either after each frame and/or after the last frame of the
Worker's list. Data produced by Workers are sent to corresponding Accumulator. In turn, accumulator sends some data to the Job,
immediately and/or when all the Workers on the worker node finish data processing. This process can be viewed as an implementation
of "map/reduce" data processing mechanism with multiple "reduce" opportunities. Users are encouraged to take every opportiunity 
to reduce data traffic by accumulating and reducing data before sending it to the Job to reduce data traffic in the system and
increase its performance.

Worker
------

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
        
    The Worker obejct is created once per
    job per worker and it persists until the worker finishes running through its set of frames. That makes it possible to
    store and accumulate some data between processing the frames. Here is an example:
    
    .. code-block:: python
    
        class Worker:
    
            Columns = ["Muon.pt"]
        
            def __init__(self, user_params, bulk_data, job, db):
                self.SumPt = 0.0
                self.NMuons = 0
        
            def frame(self, events):
                pts = events.Muon.pt
                self.NMuons += len(pts)
                self.SumPt += sum(pts)
                # do not return anything, just accumulate data from all frames
                
            def end(self):
                # return accumulated data
                return {
                    "mean_pt": self.SumPt/self.NMuons,
                    "n_muons": self.NMuons
                    }



    
Accumulator
-----------

The Framework creates one Accumulator object per worker node per user job. Accumulator's role is twofold:

 * Distribute job parameters and bulk data from the Job to local Workers
 * Collect, possibly reduce and forward data from Workers running on the same node to the Job

Accumulator is optional. If not defined, the Framework will create one to perform the data distribution and information gathering
anyway, but obviously there will be no data reduction, so all the Worker's output will be sent to the Job as is.

.. code-block:: python

    class Accumulator:
    
        def __init__(self, params, bulk_data, job, db):
            # ...
        
        def add(self, data):
            # ...
            return data_dict

        def values(self):
            # ...
            return data_dict

add() method
~~~~~~~~~~~~
Accumulator's add() method received the data dictionary returned by the Worker's frame() and end() methods, if any.
Optionally, the add() method can return some other data dictionary, or it can return None. If the add() method returns
some non-empty data dictionary, it is forwarded to the Job object.

values() method
~~~~~~~~~~~~~~~
The Framework will call Accumulator's values() method only once, when all the Workers on the worker node finish processing
their data, after calling their end() after all data returned by Worker's frame() and end() methods was passed
to the Accumulator's add() method.

The values() method returns either None or a data dictionary. This data dictionary will be sent to the Job.

Accessing Frame Data
--------------------

Frame Data Accessor
~~~~~~~~~~~~~~~~~~~

The argument of the Worker's frame() method (objects) is an Object Group Accessor object with the following attributes and methods:

**branch(barnch_name)** - method returning *Branch accessor* for the object group. Calling branch() method is equivalent to accessing the branch as if it was a property of the "objects" object:

.. code-block:: python

    def frame(self, objects):
        # ... the following are equivalent:
        b1 = objects.branch("Muon")
        b2 = objects.Muon
        
**attr(attribute_name)** - method, returns numpy array with the attribute for all the objects in the object group. Calling attr() method is equivalent to accessing the attribute as if it was a property of the "objects" object:

.. code-block:: python

    class Worker:
    
        Columns = ["event_id"]

        def frame(self, objects):
            # ... the following are equivalent:
            e1 = objects.attr("event_id")
            e2 = objects.event_id
        
**count** - attribute - returns the number of objects in the group. You can also use len(objects).

**metadata** - attribute - the frame metadata dictionary

**rgid** - attribute - returns the ID of the object group.

**filter(mask)** - method - returns an object filter object. The mask must be a single-dimension boolean (or another type convertible to boolean) numpy array with the size equal
to the number of objects. For example:

.. code-block:: python

    class Worker:
    
        Columns = ["mass"]

        def frame(self, objects):
            object_filter = objects.filter(object.mass > 4.5)

See *Filters* section below for details.

You can iterate over the Object Group Accessor object, as if it was a list of individual objects. For example:

.. code-block:: python

    class Worker:
    
        Columns = ["mass"]

        def frame(self, objects):
            for obj in objects:
                mass = obj.mass
                #...

Alternatively, individual objects can be accessed by indexing the Object Group Accessor:

.. code-block:: python

    class Worker:
    
        Columns = ["mass"]

        def frame(self, objects):
            for i in xrange(objects.count):
                mass = objects[i].mass
                #...


Branch Accessor
~~~~~~~~~~~~~~~

Calling **branch** method of the Object Group accessor object returns a Branch Accessor object. This object provides access to members of the individual branch:

**attr(attribute_name)** - method - returns numpy array with the given branch property for all the objects in the object group. Calling attr() method is equivalent to accessing the attribute as if it was a property of the branch accessor object:

.. code-block:: python

    class Worker:
    
        Columns = ["Muon.pt"]

        def frame(self, objects):
            muons = objects.Muon                    # muons is a Branch Accessor object
            # ... the following are equivalent:
            mu_pt = muons.pt
            mu_pt = muons.attr("pt")

**count** - property - returns the number of branch elements per object in the object group as an integer one-dimensional numpy array

**filter(mask)** - method - returns branch filter object. The mask argument must be a single-dimension boolean (or another type convertible to boolean) numpy array with the size equal to the total number of the branch elements in the object group. For example:

.. code-block:: python

    class Worker:
    
        Columns = ["Muon.pt"]

        def frame(self, objects):
            muon_filter = events.Muon.filter(events.Muon.pt > 300.0)
            # or...
            muons = events.Muon     # muons branch
            muon_filter = muons.filter(muons.pt > 300.0)

See *Filters* section below for details.

**pairs()** - method - creates an accessor for all combinations of branch element pairs. It is called **Combo Accessor**. 
The branch element pairs are constructed from elements of the same event only. If the event 
has 0 or 1 elements of the branch, no pairs are generated by this event. The list of generated pairs does not include swapped pairs. For example, if the event
has 3 elements of the branch, 1,2 and 3, then only 3 pairs will be generated: (1,2), (1,3) and (2,3). The list will *not* include pairs (2,1), (3,1) and (3,2).
Combo Accessor is similar to the Branch Accessor, but there are some differences. Please see below.

You can iterate over the branch accessor object, as if it was a list of individual branch elements:

.. code-block:: python

    class Worker:
    
        Columns = ["Muon.pt"]

        def frame(self, objects):
            muons = events.Muon             # branch accessor
            for mu in muons:
                mu_pt = mu.pt               # "pt" value for individual muon in the entire event group


Object Accessor
~~~~~~~~~~~~~~~


When iterating over the Object Group Accessor or applying a numeric index to it, you get an Object Accessor object:

.. code-block:: python

    class Worker:
    
        Columns = ["mass"]

        def frame(self, objects):
            for obj in objects:                 # obj is an Object Accessor
                #...


Object Accessor is used to access object attributed and branch elements associated with the object. It has the following methods and attributes:

**attr(attribute_name)** - method, returns the value of the object attribute. Calling attr() method is equivalent to accessing the attribute as if it was a property of the Object Accessor:

.. code-block:: python

    class Worker:
    
        Columns = ["mass"]

        def frame(self, objects):
            for obj in objects:                 # obj is an Object Accessor
                m1 = obj.attr("mass")           # m1 and m2 are the same
                m2 = obj.mass
                

Combo Accessor
..............

**Branch Accessor's** pairs() method returns **Combo Accessor** object. It represents all unique pairs of branch elements for all objects in the group. 
For example, let's say the group consists of 4 "objects" and each object has the folowing number of branch called "observation":

    ======== ========================
    Object    Observations
    ======== ========================
    0          2: o00, o01
    1          4: o10, o11, o12, o13
    2          1: o20
    3          3: o30, o31, o32
    ======== ========================

Then the Object Group's pairs() method will return the Combo Accessor with the following observation pairs:

    ======== ========
    Pair     Object
    ======== ========
    o00 o01   0
    o10 o11   1
    o10 o12   1
    o10 o13   1
    o11 o12   1
    o11 o13   1
    o12 o13   1
    o30 o31   3
    o30 o32   3
    o31 o32   3
    ======== ========

As you can see, the Combo Accessor includes all the pairs generated from the branch elements of the same object. The Combo Accessor can be used to iterate over 
all branch element pairs regardless of which object they belong to. For example:

.. code-block:: python

    class Worker:
    
        Columns = ["muon.p4"]

        def frame(self, objects):
            mu_pairs = events.muon.pairs()                      # this is Combo Accessor object
            for mu_pair in mu_pairs:                            # iteration produces pairs of muons for all the events in the group
                mu1, mu2 = mu_pair                              # unpack the pair
                mu_mu_mass = invariant_mass(mu1.p4, mu2.p4)     # get 4-momentums and calculate the invariant mass
                
                
You can extract first or second member of all pairs from the Combo Accessor:

.. code-block:: python

    class Worker:
    
        Columns = ["muon.p4"]

        def frame(self, objects):
            mu_pairs = events.muon.pairs()                      # this is Combo Accessor object
            mu1, mu2 = mu_pairs                                 # first and second items of each pair
            mu_mu_mass = invariant_mass_array(mu1.p4, mu2.p4)      # calculate invariant masses from vectors
            job.fill(mu_mu_mass = mu_mu_mass)
    
                

Filters
~~~~~~~

The user can filter objects and branch elements based on some boolean criteria. Filters can be applied to Object Group Accessors, Branch Accessors and
Combo Accessors. When applying a filter to these objects, the result will be the same kind of object but with reduced number of data items in it. 
There are 2 types filters - Object filters and Branch filters. Object filters are created by calling the Object Group Accessor's filter() method and
can be applied to an Object Group Accessor object. Branch filters are created by Branch Accessors and Combo Accessors and can be applied only to the
same accessor object. Filters are created by passing a boolean mask array of corresponding size to the filter() method of the accessor.


.. code-block:: python

    class Worker:
    
        Columns = ["mass","quality"]
        
        def __init__(self, params, bulk, job, db):
            self.Job = job

        def frame(self, objects):
            
            fq = objects.filter(objects.quality > 3.5)      # "object.quality > 3.5" is an expression resulting in a boolean numpy array
            good_objects = fq(objects)                      # create new Object Group Accessor with fewer objects
            
            fm = objects.filter(objects.mass > 10.3)        # another filter with another criterion
            heavy_objects = fm(objects)                     # another Object Group Accessor
            
            f_combined = fm * fq                            # filters created by the same original accessor can be combined
            f_combined = fm and fq                          # '*' and 'and' are synonyms, so are '+' and 'or'
            
            f_either_way = fm or fq                         # or'ing the filters
            heavy_or_good = f_either_way(objects)           # apply or'ed filter to the original object group

            self.Job.fill(mass_heavy = heavy_objects.mass)       # accessing "mass" attribute of filtered objects
            self.Job.fill(mass_good = good_objects.mass)

            
            # the following are errors:
            f_combined(heavy_or_good)                       # filter can be applied to its origin only
            fxyz = fm * f_either_way                        # combining filters from different origins


Branch filter examples:

.. code-block:: python

    class Worker:
    
        Columns = ["muon.pt", "muon.eta"]

        def __init__(self, params, bulk, job, db):
            self.Job = job

        def frame(self, objects):
        
            muons = objects.muon
            high_pt_filter = muon.filter(muon.pt > 100.0)
            
            # filters can be applied to both branches and arrays, so the following 2 lines produce same results:
            
            self.Job.fill(eta=high_pt_filter(muons).eta)         # filter muons, get eta's and store in histogram
            self.Job.fill(eta=high_pt_filter(muons.eta))         # get array with muon eta's, apply filter it and stote in histogram

Object filters can be converted to branch filters. This is done by replicating the object filter mask in such a way that all the branch elements of accepted
objects will be accepted, and vise versa, all the branch elements from the rejected objects will be rejected. Conversion can be done either explicitly, by
passing an existing filter to the filter() method of an accessor, or implicitly when combining filters of 2 different kinds:

.. code-block:: python

    # explicit conversion
    
    class Worker:
    
        Columns = ["mass","component.size","component.price"]

        def __init__(self, params, bulk, job, db):
            self.Job = job

        def frame(self, objects):
            
            heavy_object_filter = objects.filter(objects.mass > 10.3)
            converted_filter = objects.component.filter(heavy_object_filter)    # explicit conversion, object filter to branch filter
            
            self.Job.fill(heavy_size = converted_filter(objects.component.size))     # histogram sizes of all heavy objects
            
            # implicit conversion: combined filter is a branch filter created from object filter
            # it will accept all the components with size > 3 of all the objects with mass > 10.3
            combined_filter = heavy_object_filter * objects.component.filter(objects.component.size > 3)    
            self.Job.fill(prices_of_bulk_components_of_heavy_objcets = combined_filter(objects.component.price))
            
            
You can use filters with Combo Accessors too. Filters created by Combo Accessors are considered to be Branch Filters.

.. code-block:: python

    class Worker:
    
        Columns = ["muon.p4"]

        def __init__(self, params, bulk, job, db):
            self.Job = job

        def frame(self, events):
            mu_pairs = events.muon.pairs()                      # this is Combo Accessor object
            mu1, mu2 = mu_pairs                                 # first and second items of each pair
            
            good_pair_filter = mu_pairs.filter((mu1.pt > 100.0) * (mu2.pt > 100.0))
            good_pairs = good_pair_filter(mu_pairs)
            
            mu_mu_mass = invariant_mass_array(good_pairs[0].p4, good_pairs[1].p4)      
            self.Job.fill(mu_mu_mass = mu_mu_mass)
    
                
Job
---
Job is an object created by the framework on the client side, in the end user environment. It communicates with the framework
and its functions are:

  * Start the job
  * Monitor its progress
  * Receive and process workers output
  * Accumulate and make available histograms
  
Job is created by calling the createJob method of the Session object:

.. code-block:: python

    from striped.job import Session
    
    session = Session("config_file.yaml")
    job = session.createJob(...)
    job.run()
    
createJob() method    
~~~~~~~~~~~~~~~~~~
createJob() method has the following arguments:

.. code-block:: python

    session.createJob(
        dataset_name,
        worker_class_tag = "#__worker_class__", 
        fraction = 1.0,
        histograms = [],
        frame_selector = None,
        worker_class_source = None, 
        worker_class_file = None, 
        callbacks = None, 
        user_params = {}, 
        bulk_data = {}
    )

The parameters are:

   * dataset_name - rquired argument, a string with the name of the dataset to use
   * fraction - optional, floating point number from 0 to 1. The fraction tells the framework what is the fraction of dataset frames 
     (not the events!) the job should process. Default is to run on entire dataset.
   * histograms - a list of Hist objects to fill.
   * frame_selector - a Meta expression to select a subset of frames.
   * worker_class_source - a string with Python code with the definition of the Worker and (optionally) Accumulator classes
   * worker_class_file - local file path with Python code for Worker/Accumulator definition
   * worker_class_tag - if used in IPython/Jupyter, a commented string the user uses to tell the framework which cell has the
     Worker/Accumulator definition
   * callbacks - a list of Callback objects
   * user_params - a picklable Python object (e.g. dictionary) to pass to Worker and Accumulator as the user_params argument 
     of the constructior. This object is assumed to be small.
   * bulk_data - a single-level Python dictionary with text keys and values of integer, floating point, string or numpy ndarray type.
                

Filtering Frames by Metadata
............................
The Session.createJob() method has optional argument named "frame_selector". It can be used to pass a logical expression to the job
which will be used to select "interesting" frames only from the dataset and skip others based on the frame metadata dictionary.
To use this feature, the user needs to:

  * import Meta class definition
  * create a Meta expression
  * pass the expresion to the Session.createJob method
  
Here is an example:

.. code-block:: python

    from striped.job import Session
    from striped.common import Meta
    
    session = Session("config.yaml")
    good_frames = (Meta("E") > 100) \
        & (Meta('E') <= 300) \
        & (Meta("type") == "MC") \
        & (Meta("year") == 2016)
    job = session.createJob("my_dataset", frame_selector = good_frames)
    job.run()
    
When creating the frame selector:

  * Use Meta("key") to refer to the metadata dictionary key
  * Always use &, | instead "and", "or"

When the job runs, it will process only those frames which satisfy the expression you specify. The rest will be ignored and not
even fetched by the working, saving the job run time and the network traffic.

    



