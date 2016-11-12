package com.techspectations.viralityPredictorServer.resources;

import com.codahale.metrics.annotation.Timed;
import com.techspectations.viralityPredictorServer.ViralityPredictorApplication;
//import com.techspectations.viralityPredictorServer.core.ViralityPredictor;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Response;
import java.io.IOException;

@Path("/")
public class PredictorResource {

    @GET
    @Timed
    public double predict(@QueryParam("title") String articleTitle) {

        return ViralityPredictorApplication.viralityPredictor.getVirality(articleTitle).getVirality();
        

    }
}
