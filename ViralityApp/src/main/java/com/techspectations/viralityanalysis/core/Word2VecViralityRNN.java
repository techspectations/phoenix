package com.techspectations.viralityanalysis.core;

import com.techspectations.viralityanalysis.iterator.ViralNewsDatasetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;


public class Word2VecViralityRNN {

    public static final Logger logger = LoggerFactory.getLogger(Word2VecViralityRNN.class);

    public final String DATA_PATH;
    public  String WORD_VECTORS_PATH = "C:\\Users\\athul\\Desktop\\5man";

    private static final int BUFFER_SIZE = 4096;

    public Word2VecViralityRNN(String dataPath, String wordVectorsPath) {
        this.DATA_PATH = dataPath;
        this.WORD_VECTORS_PATH = wordVectorsPath;
    }

    public void run() throws Exception {
        int batchSize = 50;     //Number of examples in each minibatch
        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int nEpochs = 100;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 300;  //Truncate news with length (# words) greater than this
        double ratio = 0.65;

        //Set up network configuration
        int L1L2InterConnectionsCount = (int) Math.ceil(((float) vectorSize * ratio));

        MultiLayerConfiguration conf = (new NeuralNetConfiguration.Builder()).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1).updater(Updater.RMSPROP).regularization(true)
                .l2(1.0E-5D).weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0D).learningRate(0.0018D).list()
                .layer(0, (new GravesLSTM.Builder().nIn(vectorSize))
                        .nOut(L1L2InterConnectionsCount).activation("softsign").build())
                .layer(1, new RnnOutputLayer.Builder().activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nIn(L1L2InterConnectionsCount).nOut(2).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        ArrayList listeners = new ArrayList();
        listeners.add(new ScoreIterationListener(1));
        net.setListeners(listeners);

        WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false);

        System.out.println("Data Path: " + DATA_PATH);
        DataSetIterator train = new AsyncDataSetIterator(new ViralNewsDatasetIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true), 1);
        DataSetIterator test = new AsyncDataSetIterator(new ViralNewsDatasetIterator(DATA_PATH, wordVectors, 20, truncateReviewsToLength, false), 1);

        logger.info("Starting training");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(train);
            train.reset();
            logger.info("Epoch " + i + " complete. Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = new Evaluation();
            while (test.hasNext()) {
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features, false, inMask, outMask);
                evaluation.evalTimeSeries(lables, predicted, outMask);
            }
            test.reset();

            logger.info(evaluation.stats());

            File tempFile = new File("Model_"+i+"_" + String.format("%.2f", evaluation.f1()) +  ".model");
            tempFile.createNewFile();
            ModelSerializer.writeModel(net, tempFile, true);
        }

        logger.info("----- Training complete -----");
    }
}
