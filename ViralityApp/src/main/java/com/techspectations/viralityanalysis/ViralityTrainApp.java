package com.techspectations.viralityanalysis;

import com.techspectations.viralityanalysis.core.Word2VecViralityRNN;


public class ViralityTrainApp {

    //Location for the training dataset
    private static final String DATA_PATH = "C:\\Users\\athul\\Desktop\\5man";

    //Location for the Google News vectors (GoogleNews-vectors-negative300.bin)
    private static final String WORD_VECTORS_PATH = "C:\\Users\\athul\\GoogleNews-vectors-negative300.bin";

    public static void main(String[] args) throws Exception {

        Word2VecViralityRNN viralityTrainNetwork = new Word2VecViralityRNN(DATA_PATH, WORD_VECTORS_PATH);
        viralityTrainNetwork.run();
    }
}
