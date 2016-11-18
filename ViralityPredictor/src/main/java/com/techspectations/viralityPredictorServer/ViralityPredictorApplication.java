package com.techspectations.viralityPredictorServer;

import com.techspectations.viralityPredictorServer.config.ViralityPredictorConfiguration;
import com.techspectations.viralityPredictorServer.core.ViralityPredictor;
import com.techspectations.viralityPredictorServer.resources.PredictorResource;
import io.dropwizard.Application;
import io.dropwizard.setup.Bootstrap;
import io.dropwizard.setup.Environment;
import org.eclipse.jetty.servlets.CrossOriginFilter;

import javax.servlet.DispatcherType;
import javax.servlet.FilterRegistration;
import java.io.IOException;
import java.util.EnumSet;

public class ViralityPredictorApplication extends Application<ViralityPredictorConfiguration> {


    private static final String DATA_PATH = "/root/Model_14_0.57.model";

    //Location for the Google News vectors (GoogleNews-vectors-negative300.bin)
    private static final String WORD_VECTORS_PATH = "/root/GoogleNews-vectors-negative300.bin";

    public static ViralityPredictor viralityPredictor;

    public static void main(String[] args) throws Exception {
        new ViralityPredictorApplication().run(args);
    }


    @Override
    public void initialize(Bootstrap<ViralityPredictorConfiguration> bootstrap) {}

    @Override
    public void run(ViralityPredictorConfiguration configuration,
                    Environment environment) {

        ViralityPredictorConfiguration.setInstance(configuration);

        final PredictorResource predictorResource = new PredictorResource();

        try {
            viralityPredictor = new ViralityPredictor(WORD_VECTORS_PATH,DATA_PATH);
        } catch (IOException e) {
            e.printStackTrace();
        }

        environment.jersey().register(predictorResource);

        final FilterRegistration.Dynamic filter = environment.servlets().addFilter("CORS", CrossOriginFilter.class);

        filter.setInitParameter(CrossOriginFilter.ALLOWED_METHODS_PARAM, "GET,PUT,POST,OPTIONS,DELETE,HEAD");
        filter.setInitParameter(CrossOriginFilter.ALLOWED_ORIGINS_PARAM, "*");
        filter.setInitParameter(CrossOriginFilter.ALLOWED_HEADERS_PARAM, "X-Requested-With,Content-Type,Accept,Origin,authToken,authtoken");
        filter.setInitParameter(CrossOriginFilter.ALLOW_CREDENTIALS_PARAM, "true");

        filter.addMappingForUrlPatterns(EnumSet.allOf(DispatcherType.class), true, "/*");


    }

}
