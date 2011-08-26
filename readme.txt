ValueArray object
=================

The purpose of the ValueArray object is to not only have an n-dimensional array
of data but also to encode information about what a certain axis corresponds to.
For example one axis could represent a time or frequency dimension or perhaps 
repeated measurements, two of the dimensions could be matrix dimensions.

To encode this information a new tuple has been added along the regular shape tuple. 
In this implementation we call this *info*. Each element in the info tuple is an instance
of a Dim class, the Dim object contains the name, data, unit and outputformat (format string
to store format string for default formatting of the data).

The Dim object is supposed to be frozen so it can be used as a dictionary key.

Dim object
==========

ValueArray on a Dim object produces a ValueArray with info=(Dim,) and with the data from
the Dim
  >>> fi = DimSweep("f", [10, 20, 30])
  >>> a = ValueArray(fi)
  >>> a.info
  (DimSweep("f", shape=(3,)),)
  >>> a
  ValueArray([10, 20, 30])



Broadcasting
============

ValueArray objects can not be combined with regular array even if the shapes match.

The resulting shape of a broadcast between ValueArrays is determined by the union of the
info tuples, i.e. the dimensions in each array are rearranged so that the info tuples match
with size=1 dimensions where for missing Dims.

  >>> fi = DimSweep("f", [10, 20, 30])
  >>> gi = DimSweep("g", [100, 200, 300, 400])
  >>> hi = DimSweep("h", [1, 2])

  >>> a = ValueArray(zeros((3, 2)), info=(fi, hi))
  >>> b = ValueArray(zeros((2,)), info=(hi,))
  >>> c = ValueArray(zeros((3,)), info=(fi,))

  >>> (a + b).info
  (DimSweep('f', shape=(3,)), DimSweep('h', shape=(2,)))
  >>> (a + c).info
  (DimSweep('f', shape=(3,)), DimSweep('h', shape=(2,)))
  
  
This broadcasting method means that you can apply a reducing method as mean to any axis
and still be able to do meaningful operations against the original array without having to 
manually add a newaxis at the correct position.
  
However I do not want to require that the data portion of the Dim objects match only the names.
Other wise it would be very annoying to do operations like diff where we have taken unequal
subsets of arrays to do a computation e.g. a[1:] - a[:-1]. In cases like this the Dim object of
the first object is used in the result.



Indexing
========

indexing by boolean arrays have been extended so that a single dimensional array does not operate on the 
first dimension of the array but rather on the dimension with a matching Dim.

    >>> FG = ValueArray(fi) + ValueArray(gi)
    ValueArray([[110, 210, 310, 410],
           [120, 220, 320, 420],
           [130, 230, 330, 430]])
       
    >>> FG[ValueArray(fi) > 10]
    ValueArray([[120, 220, 320, 420],
           [130, 230, 330, 430]])

    >>> FG[ValueArray(gi) > 10]
    ValueArray([[310, 410],
           [320, 420],
           [330, 430]])


Array methods over axis
=======================
All array methods with an axis argument should be able to take dim objects instead of an axis index. Thus simplifying
dynamic code. You could also for instance match against the actual class of the Dim object to apply operations over 
the same kind of dimensions.

My usecase here is when doing post processing of measurement data I often have different number of repeated measurements 
for different objects::

	fi = DimSweep("freq", [10, 20, 30])
	ra = DimRep("repa", [1, 2, 3])
	rb = DimRep("repb", [1, 2, 3, 4, 5])
	obja = ValueArray(..., info=(fi, ra,))	
	objb = ValueArray(..., info=(fi, rb,))
	
	rawresult = obja - objb
	
	result = rawresult.mean(DimRep)
	s_result = rawresult.std(DimRep)
	
Here we get the mean over all repeat dimensions but leave other dimensions alone	


