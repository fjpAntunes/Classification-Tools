# coding: utf-8

# # OBIA in Python
#
# This notebook is support material for the blog post here: http://www.machinalis.com/blog/obia/
#
# The code in the blog has been simplified so it may differ from what's done here.
#

# In[1]:

# get_ipython().magic('matplotlib notebook')

import numpy as np
import os
import scipy
import sys
from matplotlib import pyplot as plt
from matplotlib import colors
from osgeo import gdal, osr, ogr
import warnings
try:
  from osgeo import ogr
  print 'Import of ogr from osgeo worked.  Hurray!\n'
except:
  print 'Import of ogr from osgeo failed\n\n'

# import ogr
# import gdal
from skimage import exposure
from skimage.exposure import rescale_intensity
from skimage.segmentation import quickshift, felzenszwalb
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# RASTER_DATA_FILE = "../rapieye_183192/2328520_2015-07-12_RE1_3A_313875_CR_browse.tif"
RASTER_DATA_FILE = "../rapieye_183192/2328520_2015-07-12_RE1_3A_313875_CR_browse.tif"
TRAIN_DATA_PATH = "../data/train/"
TEST_DATA_PATH = "../data/test/"

# In[2]:

# Testing without open
# ogr.Open()

# First, define some helper functions (taken from http://www.machinalis.com/blog/python-for-geospatial-data-processing/)

# In[3]:


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    # data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    data_source = ogr.Open(vector_data_path)
    ##
    #
    # Tem alguma coisa a ser corrigida aqui
    #
    #
    ##
    # data_source = gdal.Open(vector_data_path, GDAL_OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds

def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize all the vectors in the given directory into a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels

print("Parte 1")
# In[4]:

def getBands_data(RASTER_DATA_FILE):
    raster_dataset = gdal.Open(RASTER_DATA_FILE, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    n_bands = raster_dataset.RasterCount
    bands_data = []
    for b in range(1, n_bands + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(b for b in bands_data)
    return n_bands, bands_data, geo_transform, proj

n_bands, bands_data, geo_transform, proj = getBands_data(RASTER_DATA_FILE)
# raster_dataset = gdal.Open(RASTER_DATA_FILE, gdal.GA_ReadOnly)
# geo_transform = raster_dataset.GetGeoTransform()
# proj = raster_dataset.GetProjectionRef()
# n_bands = raster_dataset.RasterCount
# bands_data = []
# for b in range(1, n_bands + 1):
#     band = raster_dataset.GetRasterBand(b)
#     bands_data.append(band.ReadAsArray())
# bands_data = np.dstack(b for b in bands_data)

# Create images

print("Parte 2")
# In[5]:
# img = rescale_intensity(bands_data)
# Ronaldo: Acho que o indice estavam errados
# rgb_img = np.dstack([img[:, :, 3], img[:, :, 2], img[:, :, 1]])
# rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
def getImages(bands_data):
    img = rescale_intensity(bands_data)
    # rgb_img = np.dstack([img[:, :, 3], img[:, :, 2], img[:, :, 1]])
    # rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
    return img
img = getImages(bands_data)
'''
IndexError: index 3 is out of bounds for axis 1 with size 3
Indices of arrays in numpy start from 0. So an array with a second axis of 3, will be subscriptable up to a maximum index of 2.
'''
# exit(0)
# In[6]:

# plt.figure()
# plt.imshow(img)
print("Parte 3")
# In[7]:

def getSegmentsAndCmap_Quick(img):
    segments_quick = quickshift(img, kernel_size=7, max_dist=3, ratio=0.35, convert2lab=False)
    n_segments = len(np.unique(segments_quick))
    cmap_quick = colors.ListedColormap(np.random.rand(n_segments, 3))
    print(n_segments)
    return segments_quick, n_segments, cmap_quick
# segments_quick = quickshift(img, kernel_size=7, max_dist=3, ratio=0.35, convert2lab=False)
# n_segments = len(np.unique(segments_quick))
# print(n_segments)

# In[8]:
segments_quick, n_segments, cmap_quick = getSegmentsAndCmap_Quick(img)

print("Parte 4")
def plotSegmentation(segments_quick, cmap_quick):
    plt.figure()
    plt.imshow(segments_quick, interpolation='none', cmap=cmap_quick)
plotSegmentation(segments_quick, cmap_quick)
# cmap= getCmap(n_segments)
# cmap = colors.ListedColormap(np.random.rand(n_segments, 3))

# plt.figure()
# plt.imshow(segments_quick, interpolation='none', cmap=cmap_quick)

# skimage.segmentation.felzenszwalb is not prepared to work with multi-band data. So, based on their own implementation for RGB images, I apply the segmentation in each band and then combine the results. See:
# http://github.com/scikit-image/scikit-image/blob/v0.12.3/skimage/segmentation/_felzenszwalb.py#L69

# In[9]:

print("Parte 5")
def getBandSegmentation(img, n_bands):
    band_segmentation = []
    for i in range(n_bands):
        band_segmentation.append(felzenszwalb(img[:, :, i], scale=85, sigma=0.25, min_size=9))
    return band_segmentation
# band_segmentation = []
# for i in range(n_bands):
#     band_segmentation.append(felzenszwalb(img[:, :, i], scale=85, sigma=0.25, min_size=9))
band_segmentation = getBandSegmentation(img, n_bands)
# put pixels in same segment only if in the same segment in all bands. We do this by combining the band segmentation to one number

# In[10]:
print("Parte 6")
def getSegmentation(band_segmentation):
    const = [b.max() + 1 for b in band_segmentation]
    segmentation = band_segmentation[0]
    for i, s in enumerate(band_segmentation[1:]):
        segmentation += s * np.prod(const[:i + 1])
    return segmentation
segmentation = getSegmentation(band_segmentation)

# const = [b.max() + 1 for b in band_segmentation]
# segmentation = band_segmentation[0]
# for i, s in enumerate(band_segmentation[1:]):
#     segmentation += s * np.prod(const[:i + 1])


print("Parte 7")
# _, labels = np.unique(segmentation, return_inverse=True)
# segments_felz = labels.reshape(img.shape[:2])
def getSegmentsAndCmap_Felz(segmentation, img):
    _, labels = np.unique(segmentation, return_inverse=True)
    segments_felz = labels.reshape(img.shape[:2])
    cmap_felz = colors.ListedColormap(np.random.rand(len(np.unique(segments_felz)), 3))
    return segments_felz, cmap_felz
segments_felz, cmap_felz = getSegmentsAndCmap_Felz(segmentation,img)
plotSegmentation(segments_felz, cmap_felz)
print("Parte 8")
# In[11]:
# def getCmap(segments_felz):
    # cmap = colors.ListedColormap(np.random.rand(len(np.unique(segments_felz)), 3))
    # return cmap
# cmap_felz = colors.ListedColormap(np.random.rand(len(np.unique(segments_felz)), 3))
# plt.figure()
# plt.imshow(segments_felz, interpolation='none', cmap=cmap_felz)

# In[14]:
print("Parte 9")
def getCmap_QuickAndFelz(segments_quick,segments_felz):
    n_segments = max(len(np.unique(s)) for s in [segments_quick, segments_felz])
    cmap2 = colors.ListedColormap(np.random.rand(n_segments, 3))
    return cmap2
print("teste 1")
cmap_quick_felz = getCmap_QuickAndFelz(segments_quick,segments_felz)

def plotSegmentationsQuickAndFelz(img, segments_quick, segments_felz, cmap):
    # SHOW_IMAGES:
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(img, interpolation='none')
    ax1.set_title('Original image')
    ax2.imshow(segments_quick, interpolation='none', cmap=cmap)
    ax2.set_title('Quickshift segmentations')
    ax3.imshow(segments_felz, interpolation='none', cmap=cmap)
    ax3.set_title('Felzenszwalb segmentations')
    plt.show()
print("teste 1")
# plotSegmentationsQuickAndFelz(img, segments_quick, segments_felz,cmap_quick_felz)
print("teste 1")
# n_segments = max(len(np.unique(s)) for s in [segments_quick, segments_felz])
# cmap = colors.ListedColormap(np.random.rand(n_segments, 3))
# SHOW_IMAGES:
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
# ax1.imshow(img, interpolation='none')
# ax1.set_title('Original image')
# ax2.imshow(segments_quick, interpolation='none', cmap=cmap)
# ax2.set_title('Quickshift segmentations')
# ax3.imshow(segments_felz, interpolation='none', cmap=cmap)
# ax3.set_title('Felzenszwalb segmentations')
# plt.show()

# In[15]:
print("Parte 10")
# We choose the quick segmentation
def getSegmentsIdsFelz(segments_felz):
    segments = segments_felz
    segment_ids = np.unique(segments)
    print("Felzenszwalb segmentation. %i segments." % len(segment_ids))
    return segment_ids, segments
segment_ids, segments = getSegmentsIdsFelz(segments_felz)
#
# segments = segments_felz
# segment_ids = np.unique(segments)
# print("Felzenszwalb segmentation. %i segments." % len(segment_ids))

# Acho que aqui eu tenho que criar shp's para servir de conjunto de treino.

# In[16]:
print("Parte 11")

def getRowColsBandsFiles(img, TRAIN_DATA_PATH):
    rows, cols, n_bands = img.shape
    files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.shp')]
    classes_labels = [f.split('.')[0] for f in files]
    shapefiles = [os.path.join(TRAIN_DATA_PATH, f) for f in files if f.endswith('.shp')]
    print(shapefiles)
    return rows, cols, n_bands, shapefiles, classes_labels
rows, cols, n_bands, shapefiles, classes_labels = getRowColsBandsFiles(img, TRAIN_DATA_PATH)
# rows, cols, n_bands = img.shape
# files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.shp')]
# classes_labels = [f.split('.')[0] for f in files]
# shapefiles = [os.path.join(TRAIN_DATA_PATH, f) for f in files if f.endswith('.shp')]
# print(shapefiles)

print("Parte 12")
# In[17]:
def getGrountTruth(shapefiles, rows, cols, geo_transform, proj):
    ground_truth = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
    return ground_truth
# ground_truth = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
ground_truth = getGrountTruth(shapefiles, rows, cols, geo_transform, proj)
print("Parte 13")
# In[18]:

def getClasses(ground_truth):
    classes = np.unique(ground_truth)[1:]  # 0 doesn't count
    len(classes)
    return classes
# classes = np.unique(ground_truth)[1:]  # 0 doesn't count
# len(classes)
classes = getClasses(ground_truth)  # 0 doesn't count

# In[19]:
print("Parte 14")
def getSegmentsPerKlass(segments, ground_truth, classes):
    segments_per_klass = {}
    for klass in classes:
        segments_of_klass = segments[ground_truth == klass]
        segments_per_klass[klass] = set(segments_of_klass)
        print("Training segments for class %i: %i" % (klass, len(segments_per_klass[klass])))
    return segments_per_klass
# segments_per_klass = {}
# for klass in classes:
#     segments_of_klass = segments[ground_truth == klass]
#     segments_per_klass[klass] = set(segments_of_klass)
#     print("Training segments for class %i: %i" % (klass, len(segments_per_klass[klass])))
segments_per_klass = getSegmentsPerKlass(segments, ground_truth, classes)
print("Parte 15")

# In[20]:
print("Parte 16")
# ## Disambiguation
# Check if there are segments which contain training pixels of different classes.
def disambiguate(segments_per_klass):
    accum = set()
    intersection = set()
    for class_segments in segments_per_klass.values():
        intersection |= accum.intersection(class_segments)
        accum |= class_segments
    assert len(intersection) == 0
    return accum, intersection
accum, intersection = disambiguate(segments_per_klass)
# accum = set()
# intersection = set()
# for class_segments in segments_per_klass.values():
#     intersection |= accum.intersection(class_segments)
#     accum |= class_segments
# assert len(intersection) == 0
print("Parte 17")
# ### ¡No need to disambiguate!
#
# Next, we will _paint in black_ all segments that are not for training.
# The training segments will be painted of a color depending on the class.
#
# To do that we'll set as threshold the max segment id (max segments image pixel value).
# Then, to the training segments we'll assign values higher than the threshold.
# Finally, we assign 0 (zero) to pixels with values equal or below the threshold.

# In[21]:
print("Parte 18")
def train(segments, classes, segments_per_klass):
    train_img = np.copy(segments)
    threshold = train_img.max() + 1
    for klass in classes:
        klass_label = threshold + klass
        for segment_id in segments_per_klass[klass]:
            train_img[train_img == segment_id] = klass_label
    train_img[train_img <= threshold] = 0
    train_img[train_img > threshold] -= threshold
    return train_img
train_img = train(segments, classes, segments_per_klass)
# train_img = np.copy(segments)
# threshold = train_img.max() + 1
# for klass in classes:
#     klass_label = threshold + klass
#     for segment_id in segments_per_klass[klass]:
#         train_img[train_img == segment_id] = klass_label
# train_img[train_img <= threshold] = 0
# train_img[train_img > threshold] -= threshold
print("Parte 19")

# In[22]:

# Lets see the training segments
def plotTrainingSegments(train_img):
    plt.figure()
    cm = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
    cmap = colors.ListedColormap(cm)
    plt.imshow(train_img, cmap=cmap)
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5])
plotTrainingSegments(train_img)
# plt.figure()
# cm = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
# cmap = colors.ListedColormap(cm)
# plt.imshow(train_img, cmap=cmap)
# plt.colorbar(ticks=[0, 1, 2, 3, 4, 5])
print("Parte 20")

# # Training data
#
# So now, we transform each training segment into a _segment model_ and thus creating the training dataset.

# In[23]:

def segment_features(segment_pixels):
    """For each band, compute: min, max, mean, variance, skewness, kurtosis"""
    features = []
    n_pixels, n_bands = segment_pixels.shape
    for b in range(n_bands):
        stats = scipy.stats.describe(segment_pixels[:, b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if n_pixels == 1:
            # scipy.stats.describe raises a Warning and sets variance to nan
            band_stats[3] = 0.0  # Replace nan with something (zero)
        features += band_stats
    return features

# In[24]:

# ### Create all the objects:
# compute the features' vector for each segment (and append the segment ID as reference)
def createObjects(segments, segment_ids, img):
    # This is the most heavy part of the process. It could take about half an hour to finish in a not-so-fast CPU
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        objects = []
        objects_ids = []
        for segment_label in segment_ids:
            segment_pixels = img[segments == segment_label]
            segment_model = segment_features(segment_pixels)
            objects.append(segment_model)
            # Keep a reference to the segment label
            objects_ids.append(segment_label)

        print("Created %i objects" % len(objects))
        return objects, objects_ids
objects, objects_ids = createObjects(segments, segment_ids, img)
# import warnings
print("Parte 21")
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#
#     objects = []
#     objects_ids = []
#     for segment_label in segment_ids:
#         segment_pixels = img[segments == segment_label]
#         segment_model = segment_features(segment_pixels)
#         objects.append(segment_model)
#         # Keep a reference to the segment label
#         objects_ids.append(segment_label)
#
#     print("Created %i objects" % len(objects))

# ### Subset the training data


# In[25]:
print("Parte 22")
def subsetTraining(objects, objects_ids, classes, segments_per_klass):
    training_labels = []
    training_objects = []
    for klass in classes:
        class_train_objects = [v for i, v in enumerate(objects) if objects_ids[i] in segments_per_klass[klass]]
        training_labels += [klass] * len(class_train_objects)
        print("Training samples for class %i: %i" % (klass, len(class_train_objects)))
        training_objects += class_train_objects
    return training_labels, training_objects
training_labels, training_objects = subsetTraining(objects, objects_ids, classes, segments_per_klass)
# training_labels = []
# training_objects = []
# for klass in classes:
#     class_train_objects = [v for i, v in enumerate(objects) if objects_ids[i] in segments_per_klass[klass]]
#     training_labels += [klass] * len(class_train_objects)
#     print("Training samples for class %i: %i" % (klass, len(class_train_objects)))
#     training_objects += class_train_objects

# # Usupervised clustering
print("Parte 23")
# In[26]:



# In[33]:
#
# cluster = KMeans(n_clusters=5, n_jobs=-1)
#
# # In[34]:
# print("Parte 24")
# cluster.fit(objects)
#
# # In[35]:
# print("Parte 25")
# predicted = cluster.predict(objects)
#
# # In[36]:
#
# clf = np.copy(segments)
#
# # In[37]:
# print("Parte 26")
# for segment_id, klass in zip(objects_ids, predicted):
#     clf[clf == segment_id] = klass
#
# # Plotting cluster
#
# # In[38]:
#
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# # ax1.imshow(img, interpolation='none')
# ax1.set_title('Original image')
# # ax2.imshow(clf, interpolation='none', cmap=colors.ListedColormap(np.random.rand(3, 3)))
# ax2.set_title('Clasification')
# print("Parte 27")
# # Train a classifier

# In[39]:

def trainClassifier(classifierType, training_objects, training_labels):
    classifier = classifierType
    classifier.fit(training_objects, training_labels)
    return classifier
classifier = trainClassifier(RandomForestClassifier(n_jobs=-1), training_objects, training_labels)
# classifier = RandomForestClassifier(n_jobs=-1)
# # In[40]:
# classifier.fit(training_objects, training_labels)


# In[41]:
print("Parte 28")
# # Classify all segments
# Now we have to transform all segments into a _segment models_ in order to classify them
def classifySegments(classifier, objects):
    predicted = classifier.predict(objects)
    return predicted
predicted = classifySegments(classifier, objects)
# predicted = classifier.predict(objects)
# In[42]:

# # Propagate the classification
# Now that each segment has been classified, we need to propagate that classification to the pixel level. That is, given the class **k** for the segment with label **S**, generate a classification from the segmented image where all pixels in segment **S** are assigned the class **k**.
def propagateClassification(segments, objects_ids, predicted):
    clf = np.copy(segments)
    for segment_id, klass in zip(objects_ids, predicted):
        clf[clf == segment_id] = klass
    return clf
clf = propagateClassification(segments, objects_ids, predicted)

# clf = np.copy(segments)

# In[43]:
print("Parte 29")
# for segment_id, klass in zip(objects_ids, predicted):
#     clf[clf == segment_id] = klass

def plotClassification(img, clf, classes_labels):
    # plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img, interpolation='none')
    ax1.set_title('Original image')
    ax2.imshow(clf, interpolation='none', cmap=colors.ListedColormap(np.random.rand(len(classes_labels), 3)))
    ax2.set_title('Clasification')
    plt.show()
# plotClassification(img, clf, classes_labels)

'''
I Need to check why program is stopping after plot
'''
# In[46]:
# plt.figure()
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.imshow(img, interpolation='none')
# ax1.set_title('Original image')
# ax2.imshow(clf, interpolation='none', cmap=colors.ListedColormap(np.random.rand(len(classes_labels), 3)))
# ax2.set_title('Clasification')
# plt.show()
print("Parte 30")
# # Classification validation

# In[47]:

def classificationValidation(TEST_DATA_PATH, clf, classes_labels, rows, cols, geo_transform, proj):
    shapefiles = [os.path.join(TEST_DATA_PATH, "%s.shp" % c) for c in classes_labels]
    verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
    for_verification = np.nonzero(verification_pixels)
    print("Parte 31")
    # In[48]:

    verification_labels = verification_pixels[for_verification]
    predicted_labels = clf[for_verification]
    print("Parte 32")
    # In[49]:

    cm = metrics.confusion_matrix(verification_labels, predicted_labels)
    return cm, verification_labels, predicted_labels
cm, verification_labels, predicted_labels = classificationValidation(TEST_DATA_PATH, clf, classes_labels, rows, cols, geo_transform, proj)

# shapefiles = [os.path.join(TEST_DATA_PATH, "%s.shp" % c) for c in classes_labels]
# verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
# for_verification = np.nonzero(verification_pixels)
# print("Parte 31")
# # In[48]:
#
# verification_labels = verification_pixels[for_verification]
# predicted_labels = clf[for_verification]
# print("Parte 32")
# # In[49]:
#
# cm = metrics.confusion_matrix(verification_labels, predicted_labels)
# print("Parte 33")

# In[50]:

def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    # https://gist.github.com/ClementC/acf8d5f21fd91c674808
    columnwidth = max([len(x) for x in labels])
    # Print header
    # print(" " * columnwidth, end="\t")
    for label in labels:
        # print("%{0}s".format(columnwidth) % label, end="\t")
        print()
        # Print rows
    for i, label1 in enumerate(labels):
        # print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            # print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
            print()

def getClassicationMetrics(verification_labels, predicted_labels,classes_labels):
    classificationAccuracy = metrics.accuracy_score(verification_labels, predicted_labels)
    classificationReport = metrics.classification_report(verification_labels, predicted_labels,target_names=classes_labels)
    return classificationAccuracy, classificationReport

classificationAccuracy, classificationReport = getClassicationMetrics(verification_labels, predicted_labels,classes_labels)
print("Classification accuracy: %f" % classificationAccuracy)
print("Classification report:\n%s" % classificationReport)
#
#         # In[51]:
#
#         print_cm(cm, classes_labels)
#
#         # In[52]:
#
#         print("Classification accuracy: %f" %
#               metrics.accuracy_score(verification_labels, predicted_labels))
#
#         # In[53]:
#
#         print("Classification report:\n%s" %
#               metrics.classification_report(verification_labels, predicted_labels,
#                                             target_names=classes_labels))
#
#
#         # In[ ]:



