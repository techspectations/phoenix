package com.techspectations.viralityPredictorServer.model;

public class Virality {

    private final String text;
    private String output;
    private double virality;
    private int outputInt;

    public Virality(String text) {
        this.text = text;
    }

    public String getOutput() {
        return output;
    }

    public void setOutput(String output) {
        this.output = output;
    }

    public double getVirality() {
        return virality;
    }

    public void setVirality(double virality) {
        this.virality = virality;
    }

    public int getOutputInt() {
        return outputInt;
    }

    public void setOutputInt(int outputInt) {
        this.outputInt = outputInt;
    }
}
