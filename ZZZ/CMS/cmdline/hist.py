import vegascope; canvas = vegascope.LocalCanvas()
import numpy


array = numpy.random.normal(0, 1, 1000000)
histogram = Hist(bin("data", 10, -5, 5))
histogram.fill(data=array)
histogram.step("data").to(canvas)
