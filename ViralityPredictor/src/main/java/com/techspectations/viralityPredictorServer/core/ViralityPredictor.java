
package com.techspectations.viralityPredictorServer.core;

import com.techspectations.viralityPredictorServer.model.Virality;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class ViralityPredictor {

    private final WordVectors wordVectors;
    private final TokenizerFactory tokenizerFactory;
    private final int vectorSize;
    private MultiLayerNetwork network;
    private int truncateLength = 300;

    public ViralityPredictor(String wordToVectPath, String modelPath) throws IOException {
        File model = new File(modelPath);
        this.network = ModelSerializer.restoreMultiLayerNetwork(model);
        this.wordVectors = WordVectorSerializer.loadGoogleModel(new File(wordToVectPath), true, false);
        this.vectorSize = wordVectors.lookupTable().layerSize();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    public Virality getVirality(String text) {

        Virality virality = new Virality(text);

        if (text == null || text.length() == 0) return virality;

        DataSet t = getDataSet(text);
        INDArray features = t.getFeatureMatrix();
        INDArray inMask = t.getFeaturesMaskArray();
        INDArray outMask = t.getLabelsMaskArray();

        INDArray predicted = this.network.output(features, false, inMask, outMask);

        int lastElementIndex = predicted.shape()[2] - 1;
        INDArray output = predicted.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(lastElementIndex));

        double viralityProbability = output.getDouble(0);

        virality.setVirality(viralityProbability);

        virality.setOutput(viralityProbability > .65 ? "viral" : "non Viral");

        return virality;
    }

    private DataSet getDataSet(String text) {

        List<String> tokensFiltered = tokenizerFactory.create(text).getTokens().stream()
                .filter(wordVectors::hasWord).collect(Collectors.toList());

        int maxLength = tokensFiltered.size();
        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if (maxLength > truncateLength) maxLength = truncateLength;

        INDArray features = Nd4j.create(1, vectorSize, maxLength);
        INDArray labels = Nd4j.create(1, 2, maxLength);
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(1, maxLength);
        //Label masks need to be set as 1 as they are always present in the rnn time series
        INDArray labelsMask = Nd4j.zeros(1, maxLength);
        labelsMask.putScalar(new int[]{0, maxLength - 1}, 1.0);

        int[] index = new int[2];
        index[0] = 0;
        //Get word vectors for each word in review, and put them in the training data
        for (int j = 0; j < tokensFiltered.size() && j < maxLength; j++) {
            String token = tokensFiltered.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
            index[1] = j;
            featuresMask.putScalar(index, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
        }
        return new DataSet(features, labels, featuresMask, labelsMask);
    }
}

