/**
 * Wine Detector with BME688 + ESP32
 * Sends sensor data to REST API for classification
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <bsec2.h>

// ============================================
// WiFi Configuration
// ============================================
const char* ssid = "en XC";          //  !!!!! SSID !!!!!
const char* password = "necesitamosfinanciacion";  //  !!!! PASSWORD !!!!

// ============================================
// API Configuration (replace with your URL)
// ============================================
// When running locally:
const char* apiUrl = "http://192.168.0.124:5000/predict";  // !!!! PC's IP address  !!!!!

// When deployed to cloud:
// const char* apiUrl = "https://your-app-name.fly.dev/predict";

// ============================================
// BME688 Sensor Configuration
// ============================================
#define BME68X_I2C_ADDR 0x76
Bsec2 bsec;

// Delay function for BME68X
void bme68xDelayUs(uint32_t period, void *intfPtr) {
    delayMicroseconds(period);
}

// ============================================
// Timer Variables
// ============================================
unsigned long lastReading = 0;
const unsigned long readingInterval = 3000; // 3 seconds

// ============================================
// Setup
// ============================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n=================================");
    Serial.println("Wine Detector with BME688");
    Serial.println("=================================");
    
    // WiFi connection
    connectToWiFi();
    
    // Sensor initialization
    initBME688();
    
    Serial.println("\nSystem ready!");
    Serial.println("   Sending data every 3 seconds");
}

// ============================================
// WiFi Connection
// ============================================
void connectToWiFi() {
    Serial.print("\nConnecting to WiFi");
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(1000);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected!");
        Serial.print("   IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nFailed to connect to WiFi");
    }
}

// ============================================
// BME688 Initialization
// ============================================
void initBME688() {
    Serial.print("\nInitializing BME688...");
    
    if (!bsec.begin(BME68X_I2C_ADDR, Wire, bme68xDelayUs)) {
        Serial.println("Initialization failed!");
        printBsecStatus();
        while (1) delay(10);
    }
    
    // Configure desired outputs
    bsec_virtual_sensor_t sensorList[] = {
        BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_TEMPERATURE,
        BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_HUMIDITY,
        BSEC_OUTPUT_RAW_PRESSURE,
        BSEC_OUTPUT_RAW_GAS,
        BSEC_OUTPUT_IAQ
    };
    
    if (!bsec.updateSubscription(sensorList, 5, BSEC_SAMPLE_RATE_LP)) {
        Serial.println("Configuration failed!");
        printBsecStatus();
        while (1) delay(10);
    }
    
    Serial.println("BME688 ready!");
}

// ============================================
// BSEC Status Display
// ============================================
void printBsecStatus() {
    if (bsec.status < BSEC_OK) {
        Serial.println("BSEC error code: " + String(bsec.status));
    } else if (bsec.status > BSEC_OK) {
        Serial.println("BSEC warning code: " + String(bsec.status));
    }
    
    if (bsec.sensor.status < BME68X_OK) {
        Serial.println("BME68X error code: " + String(bsec.sensor.status));
    } else if (bsec.sensor.status > BME68X_OK) {
        Serial.println("BME68X warning code: " + String(bsec.sensor.status));
    }
}

// ============================================
// Send Data to API
// ============================================
void sendToAPI(float temp, float hum, float press, float gas, float iaq) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi disconnected, cannot send data");
        return;
    }
    
    HTTPClient http;
    http.begin(apiUrl);
    http.addHeader("Content-Type", "application/json");
    
    // Build JSON payload
    StaticJsonDocument<256> doc;
    doc["temperature"] = temp;
    doc["humidity"] = hum;
    doc["pressure"] = press;
    doc["gas_resistance"] = gas;
    doc["iaq"] = iaq;
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    Serial.println("\nSending to API:");
    Serial.println(jsonString);
    
    // Send POST request
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("Response received:");
        
        // Parse response JSON
        StaticJsonDocument<512> responseDoc;
        DeserializationError error = deserializeJson(responseDoc, response);
        
        if (!error) {
            const char* prediction = responseDoc["prediction"];
            float confidence = responseDoc["confidence"];
            
            Serial.print("Prediction: ");
            Serial.print(prediction);
            Serial.print(" (confidence: ");
            Serial.print(confidence * 100);
            Serial.println("%)");
            
            // Display all probabilities
            JsonObject probabilities = responseDoc["probabilities"];
            Serial.println("   Probabilities:");
            for (JsonPair kv : probabilities) {
                Serial.print("     ");
                Serial.print(kv.key().c_str());
                Serial.print(": ");
                Serial.print(kv.value().as<float>() * 100);
                Serial.println("%");
            }
        } else {
            Serial.println("   (unexpected response)");
            Serial.println(response);
        }
    } else {
        Serial.print("Error sending data: ");
        Serial.println(httpResponseCode);
    }
    
    http.end();
}

// ============================================
// Main Loop
// ============================================
void loop() {
    unsigned long now = millis();
    
    // Run BSEC algorithm
    if (bsec.run()) {
        // Display data on serial monitor
        Serial.print("Temperature: ");
        Serial.print(bsec.temperature);
        Serial.print("°C, Humidity: ");
        Serial.print(bsec.humidity);
        Serial.print("%, Gas: ");
        Serial.print(bsec.gasResistance);
        Serial.print("Ω, IAQ: ");
        Serial.println(bsec.iaq);
        
        // Send to API at specified interval
        if (now - lastReading >= readingInterval) {
            lastReading = now;
            
            sendToAPI(
                bsec.temperature,
                bsec.humidity,
                bsec.pressure,
                bsec.gasResistance,
                bsec.iaq
            );
        }
    } else {
        printBsecStatus();
    }
    
    delay(100);
}