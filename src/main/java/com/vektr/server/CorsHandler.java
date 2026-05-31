package com.vektr.server;

import io.undertow.server.HttpHandler;
import io.undertow.server.HttpServerExchange;
import io.undertow.util.HttpString;

public class CorsHandler implements HttpHandler {
    private final HttpHandler next;

    public CorsHandler(HttpHandler next) { this.next = next; }

    @Override
    public void handleRequest(HttpServerExchange ex) throws Exception {
        ex.getResponseHeaders()
            .put(new HttpString("Access-Control-Allow-Origin"), "*")
            .put(new HttpString("Access-Control-Allow-Methods"), "GET, POST, OPTIONS")
            .put(new HttpString("Access-Control-Allow-Headers"), "Content-Type");
        if (ex.getRequestMethod().toString().equals("OPTIONS")) {
            ex.setStatusCode(200); ex.endExchange(); return;
        }
        next.handleRequest(ex);
    }
}
