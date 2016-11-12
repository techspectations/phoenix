package com.techspectations.viralityPredictorServer.config;

import io.dropwizard.Configuration;

public class ViralityPredictorConfiguration extends Configuration {
    private static ViralityPredictorConfiguration instance;
    int port;

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public static ViralityPredictorConfiguration getInstance() {
        return instance;
    }

    public static void setInstance(ViralityPredictorConfiguration instance) {
        ViralityPredictorConfiguration.instance = instance;
    }
}
