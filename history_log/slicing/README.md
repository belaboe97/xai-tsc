# Latest version commit -c421171

# Restore this version in order to get the notebooks running / newer version do not include same functions

Basic idea was to use a attribution based slicing method in order to use a basic second classification head as explanation module.
Although some promising results, the problem orinated from balancing the classes. So, for example, a partitioning of a timeseries like Coffee into 5 equally spaced parts,
resulted in only 3 classes with highest mean attribution. Therefore the classes for the classification collapsed into three classes, which a very unprecise.
I concluded that other mappings might be better suited for the classification
