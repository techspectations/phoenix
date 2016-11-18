package com.techspectations.viralityanalysis.iterator;

import com.google.common.collect.Lists;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;



public class ViralNewsDatasetIterator implements DataSetIterator {

    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private final String[] viralNews;
    private final String[] nonViralNews;

    private final TokenizerFactory tokenizerFactory;
    private int cursor = 0;


    public ViralNewsDatasetIterator(String dataDirectory, WordVectors wordVectors, int batchSize, int truncateTweetsToLength, boolean train) throws IOException {

        this.batchSize = batchSize;
        this.vectorSize = wordVectors.lookupTable().layerSize();
        this.truncateLength = truncateTweetsToLength;

        //Split the data into train and test folder and place them in the data directory
        Reader csvReader = new FileReader(dataDirectory +  "\\" + (train ? "train" : "test") + "\\" + "40kTab.tsv");

        List<CSVRecord> newsList = Lists.newArrayList(CSVFormat.newFormat('\t').parse(csvReader));

        viralNews = newsList.stream().filter(e -> Integer.parseInt(e.get(1)) > 1500)
                .map(e -> e.get(0).toString())
                .collect(Collectors.toList()).toArray(new String[0]);

        nonViralNews = newsList.stream().filter(e -> Integer.parseInt(e.get(1)) < 1500)
                .map(e -> e.get(0).toString())
                .collect(Collectors.toList()).toArray(new String[0]);



        System.out.println("---------------------------------------------");
        System.out.println("ViralCount = " + viralNews.length);

        System.out.println("NonViralCount = " + nonViralNews.length);

        System.out.println("---------------------------------------------");
        this.wordVectors = wordVectors;
        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

    }


    @Override
    public DataSet next(int num) {
        if (cursor >= viralNews.length + nonViralNews.length) throw new NoSuchElementException();
        try {
            return nextDataSet(num);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {

        int trainingCount = Math.min(viralNews.length, nonViralNews.length) *2;
        //First: load tweets to String. Cycle tweet polarity
        List<String> newsList = new ArrayList<>(num);
        int[] virality = new int[num];
        for (int i = 0; i < num && cursor < trainingCount ; i++) {
            if (cursor % 2 == 0 ) {
                int viralNumber = cursor / 2;
                String news = viralNews[viralNumber];
                newsList.add(news);
                virality[i] = 1;
            } else {
                int nonViralNumber = cursor / 2;
                String tweet= nonViralNews[nonViralNumber];
                newsList.add(tweet);
                virality[i] = -1;
            }
            cursor++;
        }

        //Second: tokenize tweets and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(newsList.size());
        int maxLength = 0;
        for (String s : newsList) {
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength, tokensFiltered.size());
        }

        //Create data for training
        //Here: we have tweets.size() examples of varying lengths
        INDArray features = Nd4j.create(newsList.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(newsList.size(), 2, maxLength);    //Two labels: viral and non viral

        //Because we are dealing with tweets of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(newsList.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(newsList.size(), maxLength);

        int[] index = new int[2];
        for (int i = 0; i < newsList.size(); i++) {
            List<String> tokens = allTokens.get(i);
            index[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for (int j = 0; j < tokens.size() && j < maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
                index[1] = j;
                featuresMask.putScalar(index, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }
            int idx;
            if(virality[i] == 1) {
                idx = 0;
            } else {
                idx = 1;
            }

            int lastIdx = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);
            labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);
        }

        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    @Override
    public int totalExamples() {
        return viralNews.length + nonViralNews.length ;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("viral", "nonViral");
    }

    @Override
    public boolean hasNext() {
        return cursor < Math.min(viralNews.length, nonViralNews.length) *2;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {}

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }


    /** Convenience method to get label for review */
    public boolean isViralNews(int index) {
        return index % 2 == 0;
    }
}
