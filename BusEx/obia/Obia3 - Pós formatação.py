try:
    import numpy as np
    import os
    import scipy
    import sys
    from matplotlib import pyplot as plt
    from matplotlib import colors
    from osgeo import gdal, osr, ogr
    import warnings
    from osgeo import ogr
    # import ogr
    # import gdal
    from skimage import exposure
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import quickshift, felzenszwalb
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    print('All import worked.  Hurray!\n')
except:
    print('Some import failed\n\n')


class Obia3(object):

    def __init__(self):
        print("Funcionando")

    # Importante function, first you need to set the parameters here
    def setParameters(self):
        # Data files
        self.RASTER_DATA_FILE = "../rapieye_183192/2328520_2015-07-12_RE1_3A_313875_CR_browse.tif"
        self.TRAIN_DATA_PATH = "../data/train/"
        self.TEST_DATA_PATH = "../data/test/"
        self.CLASSIFICATED_IMAGE_PATH = "../data/classifiedImage.tiff"

        # Ploting option
        self.doPlotQuickSegmentation = False
        self.doPlotFelzSegmentation = False
        self.doPlotTrainingSegments = False
        self.doPlotClassification = True

        # Segmentation methodo selection
        self.segmentationMethod = "Quick"
        # self.segmentationMethod = "Felz"

        # Classifier selection
        self.selectedClassifier = RandomForestClassifier(n_jobs=-1)

        # Printing options
        self.printCm = False
        self.printClassificationMetrics = True

    def setParametersQGIS(self, image_path, classified_image_path, show_output, classifierType ,classifier, segmenter, selected_clf_layers=None,selected_valitation_layers=None):
        # Data files
        self.RASTER_DATA_FILE = image_path
        self.TRAIN_DATA_PATH = selected_clf_layers
        # self.TRAIN_DATA_PATH = "../data/train/"
        # self.TEST_DATA_PATH = "../data/test/"
        self.TEST_DATA_PATH = selected_valitation_layers
        self.CLASSIFICATED_IMAGE_PATH = classified_image_path

        self.showOutput = show_output

        # Ploting option
        self.doPlotClassification = show_output

        # Segmentation methodo selection
        self.segmentationMethod = segmenter
        # self.segmentationMethod = "Felz"

        # Classifier selection
        self.classifiers = {
            'Neural Net': MLPClassifier(alpha=1),
            'Linear SVM': SVC(kernel="linear", C=0.025),
            'RBF SVM': SVC(gamma=2, C=1),
            'Gaussian Process': GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5),
            'Random Forest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            'Nearest Neighbors': KNeighborsClassifier(3),
            'AdaBoost' : AdaBoostClassifier(),
            'Naive Bayes': GaussianNB(),
            'Quadratic Discrimant': QuadraticDiscriminantAnalysis}

        teste = True
        if classifierType == "Supervisionado":
            if teste:
                self.selectedClassifier = self.classifiers[classifier]
            else:
                if classifier =="Random Florest":
                    self.selectedClassifier = RandomForestClassifier(n_jobs=-1)
        elif classifierType == "Nao supervisionado":
            pass

    def create_mask_from_vector(self, vector_data_path, cols, rows, geo_transform, projection, target_value=1):
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

    def vectors_to_raster(self, file_paths, rows, cols, geo_transform, projection):
        """Rasterize all the vectors in the given directory into a single image."""
        labeled_pixels = np.zeros((rows, cols))
        for i, path in enumerate(file_paths):
            label = i + 1
            ds = self.create_mask_from_vector(path, cols, rows, geo_transform,
                                         projection, target_value=label)
            band = ds.GetRasterBand(1)
            labeled_pixels += band.ReadAsArray()
            ds = None
        return labeled_pixels

    def getBands_data(self, RASTER_DATA_FILE):
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

    def raster_to_tiff(self,filename, rows, cols,geo_transform, projection):
        pass

    def raster_to_tiff2(self, img_array, path, rows, cols,geo_transform, projection):
        """Array > Raster
        Save a raster from a C order array.

        :param array: ndarray
        """
        # dst_filename = '/a_file/name.tiff'

        # You need to get those values like you did.
        x_pixels = 16  # number of pixels in x
        y_pixels = 16  # number of pixels in y
        PIXEL_SIZE = 3  # size of the pixel...
        x_min = 553648
        y_max = 7784555  # x_min & y_max are like the "top left" corner.
        wkt_projection = 'a projection in wkt that you got from other file'

        driver = gdal.GetDriverByName('GTiff')

        dataset = driver.Create(path,cols,rows,1,gdal.GDT_Float32,)

        dataset.SetGeoTransform(geo_transform)

        dataset.SetProjection(projection)
        dataset.GetRasterBand(1).WriteArray(img_array)
        dataset.FlushCache()  # Write to disk.
        return dataset, dataset.GetRasterBand(1)  # If you need to return, remenber to return  also the dataset because the band don`t live without dataset.

    # Create images
    def getImages(self, bands_data):
        img = rescale_intensity(bands_data)
        # rgb_img = np.dstack([img[:, :, 3], img[:, :, 2], img[:, :, 1]])
        # rgb_img = np.dstack([img[:, :, 2], img[:, :, 1], img[:, :, 0]])
        return img

    def getSegmentsCmapID_Quick(self, img):
        segments_quick = quickshift(img, kernel_size=7, max_dist=3, ratio=0.35, convert2lab=False)
        segment_quick_ids = np.unique(segments_quick)
        n_segments = len(np.unique(segments_quick))
        cmap_quick = colors.ListedColormap(np.random.rand(n_segments, 3))
        print(n_segments)
        return segments_quick, n_segments, cmap_quick, segment_quick_ids

    def plotSegmentation(self, segments_quick, cmap_quick):
        plt.figure()
        plt.imshow(segments_quick, interpolation='none', cmap=cmap_quick)

    # Segments the band segmentation of the image
    def getBandSegmentation(self,img, n_bands):
        band_segmentation = []
        for i in range(n_bands):
            band_segmentation.append(felzenszwalb(img[:, :, i], scale=85, sigma=0.25, min_size=9))
        return band_segmentation

    def getSegmentation(self, band_segmentation):
        const = [b.max() + 1 for b in band_segmentation]
        segmentation = band_segmentation[0]
        for i, s in enumerate(band_segmentation[1:]):
            segmentation += s * np.prod(const[:i + 1])
        return segmentation

    def getSegmentsCmapIds_Felz(self, segmentation, img):
        _, labels = np.unique(segmentation, return_inverse=True)
        segments_felz = labels.reshape(img.shape[:2])
        cmap_felz = colors.ListedColormap(np.random.rand(len(np.unique(segments_felz)), 3))
        segments = segments_felz
        segment_felz_ids = np.unique(segments)
        print("Felzenszwalb segmentation. %i segments." % len(segment_felz_ids))
        return segments_felz, cmap_felz, segment_felz_ids

    def getCmap_QuickAndFelz(self,segments_quick, segments_felz):
        n_segments = max(len(np.unique(s)) for s in [segments_quick, segments_felz])
        cmap2 = colors.ListedColormap(np.random.rand(n_segments, 3))
        return cmap2

    # SHOW_IMAGES:
    def plotSegmentationsQuickAndFelz(self,img, segments_quick, segments_felz, cmap):
        # SHOW_IMAGES:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        ax1.imshow(img, interpolation='none')
        ax1.set_title('Original image')
        ax2.imshow(segments_quick, interpolation='none', cmap=cmap)
        ax2.set_title('Quickshift segmentations')
        ax3.imshow(segments_felz, interpolation='none', cmap=cmap)
        ax3.set_title('Felzenszwalb segmentations')
        plt.show()

    # We choose the quick segmentation
    def getSegmentsIdsFelz(self,segments_felz):
        segments = segments_felz
        segment_felz_ids = np.unique(segments)
        print("Felzenszwalb segmentation. %i segments." % len(segment_felz_ids))
        return segment_felz_ids

    def getRowColsBandsFiles(self,img, TRAIN_DATA_PATH):
        rows, cols, n_bands = img.shape
        files = [f for f in os.listdir(TRAIN_DATA_PATH) if f.endswith('.shp')]
        classes_labels = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(TRAIN_DATA_PATH, f) for f in files if f.endswith('.shp')]
        print(shapefiles)
        return rows, cols, n_bands, shapefiles, classes_labels

    def QGISgetRowColsBandsFiles(self,img, TRAIN_DATA_PATH):
        rows, cols, n_bands = img.shape
        # files = TRAIN_DATA_PATH.values()
        # classes_labels = TRAIN_DATA_PATH.keys()
        classes_labels = TRAIN_DATA_PATH.keys()
        shapefiles = TRAIN_DATA_PATH.values()
        # shapefiles = TRAIN_DATA_PATH.values()
        print(shapefiles)
        return rows, cols, n_bands, shapefiles, classes_labels

    def getGrountTruth(self, shapefiles, rows, cols, geo_transform, proj):
        ground_truth = self.vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
        # print(ground_truth.__class__.__name__)
        return ground_truth

    def getClasses(self, ground_truth):
        classes = np.unique(ground_truth)[1:]  # 0 doesn't count
        len(classes)
        return classes

    def getSegmentsPerKlass(self, segments, ground_truth, classes):
        segments_per_klass = {}
        for klass in classes:
            segments_of_klass = segments[ground_truth == klass]
            segments_per_klass[klass] = set(segments_of_klass)
            print("Training segments for class %i: %i" % (klass, len(segments_per_klass[klass])))
        return segments_per_klass

    # ## Disambiguation
    # Check if there are segments which contain training pixels of different classes.
    def disambiguate(self, segments_per_klass):
        accum = set()
        intersection = set()
        for class_segments in segments_per_klass.values():
            intersection |= accum.intersection(class_segments)
            accum |= class_segments
        assert len(intersection) == 0
        return accum, intersection

    # Next, we will _paint in black_ all segments that are not for training.
    # The training segments will be painted of a color depending on the class.
    #
    # To do that we'll set as threshold the max segment id (max segments image pixel value).
    # Then, to the training segments we'll assign values higher than the threshold.
    # Finally, we assign 0 (zero) to pixels with values equal or below the threshold.
    def train(self, segments, classes, segments_per_klass):
        train_img = np.copy(segments)
        threshold = train_img.max() + 1
        for klass in classes:
            klass_label = threshold + klass
            for segment_id in segments_per_klass[klass]:
                train_img[train_img == segment_id] = klass_label
        train_img[train_img <= threshold] = 0
        train_img[train_img > threshold] -= threshold
        return train_img

    # Lets see the training segments
    def plotTrainingSegments(self, train_img):
        plt.figure()
        cm = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1]])
        cmap = colors.ListedColormap(cm)
        plt.imshow(train_img, cmap=cmap)
        plt.colorbar(ticks=[0, 1, 2, 3, 4, 5])

    # # Training data
    # So now, we transform each training segment into a _segment model_ and thus creating the training dataset.
    def segment_features(self, segment_pixels):
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

    # ### Create all the objects:
    # compute the features' vector for each segment (and append the segment ID as reference)
    # This is the most heavy part of the process. It could take about half an hour to finish in a not-so-fast CPU
    def createObjects(self, segments, segment_ids, img):
        # This is the most heavy part of the process. It could take about half an hour to finish in a not-so-fast CPU
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            objects = []
            objects_ids = []
            for segment_label in segment_ids:
                segment_pixels = img[segments == segment_label]
                segment_model = self.segment_features(segment_pixels)
                objects.append(segment_model)
                # Keep a reference to the segment label
                objects_ids.append(segment_label)

            print("Created %i objects" % len(objects))
            return objects, objects_ids

    # Subset the training data
    def subsetTraining(self, objects, objects_ids, classes, segments_per_klass):
        training_labels = []
        training_objects = []
        for klass in classes:
            class_train_objects = [v for i, v in enumerate(objects) if objects_ids[i] in segments_per_klass[klass]]
            training_labels += [klass] * len(class_train_objects)
            print("Training samples for class %i: %i" % (klass, len(class_train_objects)))
            training_objects += class_train_objects
        return training_labels, training_objects

    # Train a classifier
    def trainClassifier(self, classifierType, training_objects, training_labels):
        classifier = classifierType
        classifier.fit(training_objects, training_labels)
        return classifier

    # # Classify all segments
    # Now we have to transform all segments into a _segment models_ in order to classify them
    def classifySegments(self, classifier, objects):
        predicted = classifier.predict(objects)
        return predicted

    # # Propagate the classification
    # Now that each segment has been classified, we need to propagate that classification to the pixel level. That is, given the class **k** for the segment with label **S**, generate a classification from the segmented image where all pixels in segment **S** are assigned the class **k**.
    def propagateClassification(self, segments, objects_ids, predicted):
        clf = np.copy(segments)
        for segment_id, klass in zip(objects_ids, predicted):
            clf[clf == segment_id] = klass
        return clf

    def plotClassification(self, img, clf, classes_labels):
        # plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(img, interpolation='none')
        ax1.set_title('Original image')
        # ax2.imshow(clf, interpolation='none')
        ax2.imshow(clf, interpolation='none', cmap=colors.ListedColormap(np.random.rand(len(classes_labels), 3)))
        ax2.set_title('Clasification')
        plt.show()

    # # Classification validation - Confusion Matrix
    def classificationValidation(self,TEST_DATA_PATH, clf, classes_labels, rows, cols, geo_transform, proj):
        # shapefiles = [os.path.join(TEST_DATA_PATH, "%s.shp" % c) for c in classes_labels]
        shapefiles = TEST_DATA_PATH.values()
        verification_pixels = self.vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
        for_verification = np.nonzero(verification_pixels)
        print("Parte 31")
        # In[48]:

        verification_labels = verification_pixels[for_verification]
        predicted_labels = clf[for_verification]
        print("Parte 32")
        # In[49]:

        cm = metrics.confusion_matrix(verification_labels, predicted_labels)
        return cm, verification_labels, predicted_labels

    # Print the confusion matrix
    def print_cm(self,cm, labels):
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

    # Return classification metrics: classificationAccuracy, classificationReport
    def getClassicationMetrics(self, verification_labels, predicted_labels, classes_labels):
        classificationAccuracy = metrics.accuracy_score(verification_labels, predicted_labels)
        classificationReport = metrics.classification_report(verification_labels, predicted_labels,
                                                             target_names=classes_labels)
        return classificationAccuracy, classificationReport

    def QgisClassification(self):
        # Segment Original Image into Bands
        n_bands, bands_data, self.geo_transform, self.proj = self.getBands_data(self.RASTER_DATA_FILE)
        img = self.getImages(bands_data)
        band_segmentation = self.getBandSegmentation(img, n_bands)
        segmentation = self.getSegmentation(band_segmentation)

        # Choose the type of segmentation
        if self.segmentationMethod == "Quick":
            segments_quick, n_segments, cmap_quick, segment_quick_id = self.getSegmentsCmapID_Quick(img)
            segments = segments_quick
            segment_ids = segment_quick_id
        elif self.segmentationMethod == "Felz":
            segments_felz, cmap_felz, segment_felz_id = self.getSegmentsCmapIds_Felz(segmentation, img)
            segments = segments_felz
            segment_ids = segment_felz_id


        # Acho que aqui eu tenho que criar shp's para servir de conjunto de treino.
        self.rows, self.cols, n_bands, shapefiles, self.classes_labels = self.QGISgetRowColsBandsFiles(img, self.TRAIN_DATA_PATH)
        ground_truth = self.getGrountTruth(shapefiles, self.rows, self.cols, self.geo_transform, self.proj)
        classes = self.getClasses(ground_truth)  # 0 doesn't count
        segments_per_klass = self.getSegmentsPerKlass(segments, ground_truth, classes)

        # Creating objects
        objects, objects_ids = self.createObjects(segments, segment_ids, img)
        training_labels, training_objects = self.subsetTraining(objects, objects_ids, classes, segments_per_klass)

        #Training the classifier
        classifier = self.trainClassifier(self.selectedClassifier, training_objects, training_labels)

        # Classify all segments
        predicted = self.classifySegments(classifier, objects)

        # Propagate Classification
        self.clf_image = self.propagateClassification(segments, objects_ids, predicted)

        # Save classified image to tiff
        self.raster_to_tiff2(self.clf_image, self.CLASSIFICATED_IMAGE_PATH, self.rows, self.cols, self.geo_transform,
                             self.proj)

        # if self.TEST_DATA_PATH is not None:
        #     return self.validateClassification()

    def validateClassification(self):
        # Get Classification Confusion Matrix
        cm, verification_labels, predicted_labels = self.classificationValidation(self.TEST_DATA_PATH, self.clf_image,
                                                                                  self.classes_labels, self.rows, self.cols,
                                                                                  self.geo_transform, self.proj)
        # if self.printCm == True:
        #     self.print_cm(cm, classes_labels)

        # Get Classification Metrics
        classificationAccuracy, classificationReport = self.getClassicationMetrics(verification_labels,
                                                                                   predicted_labels, self.classes_labels)
        # if self.printClassificationMetrics == True:
        #     print("Classification accuracy: %f" % classificationAccuracy)
        #     print("Classification report:\n%s" % classificationReport)
        return cm, classificationAccuracy, classificationReport

    def AllProcess(self):
        self.setParameters()

        # Segment Original Image into Bands
        n_bands, bands_data, self.geo_transform, self.proj = self.getBands_data(self.RASTER_DATA_FILE)
        img = self.getImages(bands_data)
        band_segmentation = self.getBandSegmentation(img, n_bands)
        segmentation = self.getSegmentation(band_segmentation)

        # Quick Segmentation
        segments_quick, n_segments, cmap_quick, segment_quick_id = self.getSegmentsCmapID_Quick(img)
        if self.doPlotQuickSegmentation == True:
            self.plotSegmentation(segments_quick, cmap_quick)


        #Felz Segmentation
        segments_felz, cmap_felz, segment_felz_id = self.getSegmentsCmapIds_Felz(segmentation, img)
        if self.doPlotFelzSegmentation == True:
            self.plotSegmentation(segments_felz, cmap_felz)

        # Felz and Quick Segmentation
        cmap_quick_felz = self.getCmap_QuickAndFelz(segments_quick, segments_felz)

        # Choose the type of segmentation
        if self.segmentationMethod == "Quick":
            segments = segments_quick
            segment_ids = segment_quick_id
        elif self.segmentationMethod == "Felz":
            segments = segments_felz
            segment_ids = segment_felz_id


        # Acho que aqui eu tenho que criar shp's para servir de conjunto de treino.
        self.rows, self.cols, n_bands, shapefiles, classes_labels = self.getRowColsBandsFiles(img, self.TRAIN_DATA_PATH)
        ground_truth = self.getGrountTruth(shapefiles, self.rows, self.cols, self.geo_transform, self.proj)
        classes = self.getClasses(ground_truth)  # 0 doesn't count
        segments_per_klass = self.getSegmentsPerKlass(segments, ground_truth, classes)
        accum, intersection = self.disambiguate(segments_per_klass)

        # Training the image
        train_img = self.train(segments, classes, segments_per_klass)
        if self.doPlotTrainingSegments == True:
            self.plotTrainingSegments(train_img)

        # Creating objects
        objects, objects_ids = self.createObjects(segments, segment_ids, img)
        training_labels, training_objects = self.subsetTraining(objects, objects_ids, classes, segments_per_klass)

        #Training the classifier
        classifier = self.trainClassifier(self.selectedClassifier, training_objects, training_labels)

        # Classify all segments
        predicted = self.classifySegments(classifier, objects)

        # Propagate Classification
        clf_image = self.propagateClassification(segments, objects_ids, predicted)

        # Save classified image to tiff
        self.raster_to_tiff2(clf_image, self.CLASSIFICATED_IMAGE_PATH, self.rows, self.cols, self.geo_transform,
                             self.proj)

        # Plot Classification
        if self.doPlotClassification == True:
            print(img.__class__.__name__)
            self.plotClassification(img, clf_image, classes_labels)

        # Get Classification Confusion Matrix
        cm, verification_labels, predicted_labels = self.classificationValidation(self.TEST_DATA_PATH, clf_image, classes_labels, self.rows,self.cols, self.geo_transform, self.proj)
        if self.printCm == True:
            self.print_cm(cm, classes_labels)

        # Get Classification Metrics
        classificationAccuracy, classificationReport = self.getClassicationMetrics(verification_labels, predicted_labels,classes_labels)
        if self.printClassificationMetrics ==True:
            print("Classification accuracy: %f" % classificationAccuracy)
            print("Classification report:\n%s" % classificationReport)


if __name__ == '__main__':
    myObiaTeste = Obia3()
    myObiaTeste.AllProcess()