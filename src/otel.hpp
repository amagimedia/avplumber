#pragma once

// OpenTelemetry entry point for the engine. The metrics work lands later; this is
// the configuration seam it will plug into, wired now so enabling the SDK is a
// build flag (HAVE_OTEL=1) rather than a source change.
//
// Default build (HAVE_OTEL undefined): init() is a no-op and nothing from the
// OpenTelemetry headers is included, so the dependency stays optional.
//
// Log lines are NOT routed through OpenTelemetry: the engine's logstream output is
// AVP_LOG_FORMAT-controlled text or OTel JSON on stderr (src/util.cpp), and the
// standard forbids non-JSON lines on stdout. Only metrics (and later traces, if any)
// go through the SDK, and only to the configured endpoint -- never to stdout.

#ifndef HAVE_OTEL

namespace avplumber_otel {

inline void init() {
}

}  // namespace avplumber_otel

#else

#include <opentelemetry/exporters/ostream/metric_exporter_factory.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_options.h>
#include <opentelemetry/sdk/metrics/meter_provider_factory.h>
#include <opentelemetry/sdk/metrics/view/view_registry_factory.h>
#include <opentelemetry/sdk/resource/resource.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "app_version.hpp"

namespace avplumber_otel {

inline std::string env_or(const char *primary, const char *secondary, const char *fallback) {
    const char *value = getenv(primary);
    if (value && *value) return {value};
    value = getenv(secondary);
    if (value && *value) return {value};
    return {fallback};
}

// The service identity matches the log lines (the JSON form of logstream output
// carries the same three fields).
inline opentelemetry::sdk::resource::Resource resource() {
    return opentelemetry::sdk::resource::Resource::Create(
        {{"service.name", env_or("OTEL_SERVICE_NAME", "", "avplumber")},
         {"service.version", env_or("OTEL_SERVICE_VERSION", "", APP_VERSION)},
         {"deployment.environment", env_or("DEPLOYMENT_ENVIRONMENT", "OTEL_DEPLOYMENT_ENVIRONMENT", "unknown")}});
}

// Installs the meter provider: a periodic reader pushing to stderr's ostream
// exporter until the deployment's OTLP exporter replaces it (the exporter choice is
// the one place metrics touch stdout-adjacent streams, hence cerr, never cout).
// Reads OTEL_EXPORTER_METRIC_EXPORT_INTERVAL_MS (default 60000). Call once at
// process start, before the graph starts.
inline void init() {
    namespace metrics_sdk = opentelemetry::sdk::metrics;
    namespace metrics_ostream = opentelemetry::exporter::metrics;

    const char *interval = getenv("OTEL_EXPORTER_METRIC_EXPORT_INTERVAL_MS");
    metrics_sdk::PeriodicExportingMetricReaderOptions options;
    options.export_interval_millis = std::chrono::milliseconds(interval ? atoi(interval) : 60000);
    options.export_timeout_millis = std::chrono::milliseconds(30000);
    auto reader = metrics_sdk::PeriodicExportingMetricReaderFactory::Create(
        metrics_ostream::OStreamMetricExporterFactory::Create(std::cerr), options);
    auto provider = metrics_sdk::MeterProviderFactory::Create(
        metrics_sdk::ViewRegistryFactory::Create(), resource());
    provider->AddMetricReader(std::shared_ptr<metrics_sdk::MetricReader>(reader.release()));
    std::shared_ptr<opentelemetry::metrics::MeterProvider> shared(std::move(provider));
    opentelemetry::metrics::Provider::SetMeterProvider(shared);
}

}  // namespace avplumber_otel

#endif