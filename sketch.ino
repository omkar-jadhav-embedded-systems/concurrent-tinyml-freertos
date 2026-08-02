// ====================================================================
//  FINAL VERSION: TINYML ANOMALY + BATTERY SOC/SOH ON FREERTOS
// ====================================================================

#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <math.h>

// Include BOTH models
#include "anomaly_model.h" 
#include "soc_soh.h" 

const int PIN_LED = 13;

// --- FreeRTOS Mutex ---
SemaphoreHandle_t tflmMutex;

// =====================================================================
//                       DSP FILTER CONFIGURATIONS
// =====================================================================
const int FILTER_WINDOW_SIZE = 5;

// Anomaly DSP (12V Domain)
float anom_voltageHistory[FILTER_WINDOW_SIZE];
float anom_currentHistory[FILTER_WINDOW_SIZE];
float anom_tempHistory[FILTER_WINDOW_SIZE];
int anom_filterIndex = 0;

// Battery DSP (4V Domain)
float batt_voltageHistory[FILTER_WINDOW_SIZE];
float batt_currentHistory[FILTER_WINDOW_SIZE];
float batt_tempHistory[FILTER_WINDOW_SIZE];
int batt_filterIndex = 0;

// =====================================================================
//                       TFLM GLOBALS & SCALING
// =====================================================================
namespace {
  // --- Anomaly Model Globals ---
  const tflite::Model* anom_model = nullptr;
  tflite::MicroInterpreter* anom_interpreter = nullptr;
  TfLiteTensor* anom_input = nullptr;
  TfLiteTensor* anom_output = nullptr;
  constexpr int kAnomArenaSize = 4 * 1024;
  uint8_t anom_tensor_arena[kAnomArenaSize];

  // --- Battery Model Globals ---
  const tflite::Model* batt_model = nullptr;
  tflite::MicroInterpreter* batt_interpreter = nullptr;
  TfLiteTensor* batt_input = nullptr;
  TfLiteTensor* batt_output = nullptr;
  constexpr int kBattArenaSize = 12 * 1024; 
  uint8_t batt_tensor_arena[kBattArenaSize];
}

// Anomaly Scaling
const float ANOM_VOLTAGE_MIN = 11.5034;
const float ANOM_VOLTAGE_MAX = 13.4998;
const float ANOM_CURRENT_MIN = 1.0003;
const float ANOM_CURRENT_MAX = 7.4999;
const float ANOM_TEMP_MIN = 20.0101;
const float ANOM_TEMP_MAX = 84.9987;
const float ANOMALY_THRESHOLD = 0.0418;

// Battery Scaling
const float BATT_VOLTAGE_MIN = 3.0039;
const float BATT_VOLTAGE_MAX = 4.1997;
const float BATT_CURRENT_MIN = -4.9999;
const float BATT_CURRENT_MAX = 4.9956;
const float BATT_TEMP_MIN    = -9.9982;
const float BATT_TEMP_MAX    = 49.9610;

// --- Function Prototypes ---
void TaskAnomaly(void *pvParameters);
void TaskBattery(void *pvParameters);
void simulate_battery_scenario(float target_v, float target_a, float target_t);
void run_anomaly_inference(float raw_data[3]);

// =====================================================================
//                             SETUP
// =====================================================================
void setup() 
{
  Serial.begin(115200);
  while (!Serial && millis() < 2000);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  // 1. Initialize FreeRTOS Mutex
  tflmMutex = xSemaphoreCreateMutex();
  if (tflmMutex == NULL) {
    Serial.println("FATAL: Mutex creation failed!");
    while(1);
  }

  // 2. Initialize Anomaly Model
  anom_model = tflite::GetModel(anomaly_model);
  static tflite::MicroMutableOpResolver<4> anom_op_resolver;
  anom_op_resolver.AddFullyConnected(); 
  anom_op_resolver.AddRelu(); 
  anom_op_resolver.AddQuantize(); 
  anom_op_resolver.AddDequantize();
  static tflite::MicroInterpreter anom_static_interpreter(anom_model, anom_op_resolver, anom_tensor_arena, kAnomArenaSize);
  anom_interpreter = &anom_static_interpreter;
  anom_interpreter->AllocateTensors();
  anom_input = anom_interpreter->input(0);
  anom_output = anom_interpreter->output(0);

  // 3. Initialize Battery Model
  batt_model = tflite::GetModel(battery_model); 
  static tflite::MicroMutableOpResolver<6> batt_op_resolver;
  batt_op_resolver.AddFullyConnected(); 
  batt_op_resolver.AddRelu(); 
  batt_op_resolver.AddQuantize(); 
  batt_op_resolver.AddDequantize();
  batt_op_resolver.AddReshape(); 
  static tflite::MicroInterpreter batt_static_interpreter(batt_model, batt_op_resolver, batt_tensor_arena, kBattArenaSize);
  batt_interpreter = &batt_static_interpreter;
  batt_interpreter->AllocateTensors();
  batt_input = batt_interpreter->input(0);
  batt_output = batt_interpreter->output(0);

  Serial.println("\n=============================================");
  Serial.println(" TinyML Anomaly System with DSP Noise Filter ");
  Serial.println("=============================================");

  // --- PRIME THE DSP FILTER ---
  // Before starting, we must fill the filter's history with normal data
  // to establish a stable baseline and avoid initial false alarms.
  Serial.println("\n>>> Priming DSP filter with initial normal readings...");
  float priming_data[3] = {12.1, 3.5, 45.2}; // A typical normal reading
  for (int i = 0; i < FILTER_WINDOW_SIZE; i++) 
  {
      anom_voltageHistory[i] = priming_data[0];
      anom_currentHistory[i] = priming_data[1];
      anom_tempHistory[i] = priming_data[2];
  }
  Serial.println(">>> Filter is primed.\n");


  Serial.println("\n=============================================");
  Serial.println(" TinyML Dual-Model System (FreeRTOS Mutex)   ");
  Serial.println("=============================================");

  xTaskCreatePinnedToCore(TaskAnomaly, "TaskAnomaly", 4096, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(TaskBattery, "TaskBattery", 4096, NULL, 1, NULL, 1);
  
  Serial.println("\n--- SYSTEM BENCHMARK ---");
  Serial.printf("Total Free Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Largest Free Block: %d bytes\n", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
  Serial.println("------------------------\n");
}

void TaskBattery(void *pvParameters) 
{
  float test_data_1[] = {3.744087307, -0.300117318, -9.405358271};
  float test_data_2[] = {3.819954452, 3.587642793, 38.11041011};
  float test_data_3[] = {3.079982449, -3.812909045, 49.96101803};
  int step = 0;

  for (;;) 
  {  
    xSemaphoreTake(tflmMutex, portMAX_DELAY);
    Serial.println("\n--- [BATTERY TASK EXECUTING] ---");
    
    if (step == 0) 
    {
      Serial.println(">>> RUNNING TEST 1");
      simulate_battery_scenario(test_data_1);
    } 
    else if (step == 1) 
    {
      Serial.println(">>> RUNNING TEST 2");
      simulate_battery_scenario(test_data_2); 
    } 
    else if (step == 2) 
    {
      Serial.println(">>> RUNNING TEST 3");
      simulate_battery_scenario(test_data_3);
    }
        // RELEASE CPU MUTEX
    xSemaphoreGive(tflmMutex);
    step = (step + 1) % 3;
    Serial.printf("📈 Battery Task Stack High Water Mark: %u words\n", uxTaskGetStackHighWaterMark(NULL));
    vTaskDelay(pdMS_TO_TICKS(5000)); // Total delay roughly 4s cycle
  }
}

void TaskAnomaly(void *pvParameters) 
{
  float normal_data[3] = {12.12, 3.48, 45.15};
  float noisy_spike[3] = {12.10, 9.50, 45.20};
  float sustained_anomaly[3] = {12.20, 9.50, 48.00};
  int step = 0;

  for (;;) 
  {
    xSemaphoreTake(tflmMutex, portMAX_DELAY);
    Serial.println("\n--- [ANOMALY TASK EXECUTING] ---");

    if (step == 0) 
    { 
      // Normal reading
      run_anomaly_inference(normal_data); 
    }
    else if (step == 1) 
    { 
      // Single Spike (Filter absorbs it, no alarm)
      run_anomaly_inference(noisy_spike); 
    }
    else if (step == 2) 
    { 
      // Return to normal
      run_anomaly_inference(normal_data); 
    }
    else if (step >= 3) 
    { 
      Serial.println(">>> [Simulating Sustained Hardware Overcurrent]");
        
      for (int i = 0; i < FILTER_WINDOW_SIZE; i++) 
      {
        run_anomaly_inference(sustained_anomaly);
        vTaskDelay(pdMS_TO_TICKS(1000));
      }
    }
        // RELEASE CPU MUTEX
    xSemaphoreGive(tflmMutex);
    step = (step + 1) % 5;
    Serial.printf("📈 Anomaly Task Stack High Water Mark: %u words\n", uxTaskGetStackHighWaterMark(NULL));
    vTaskDelay(pdMS_TO_TICKS(1000)); // Run every 4 seconds
  }
}

float update_filter(float newValue, float history[]) 
{
  history[anom_filterIndex] = newValue; // Insert new value
  float sum = 0;
  for (int i = 0; i < FILTER_WINDOW_SIZE; i++) 
  { 
    sum += history[i]; 
  }
  return sum / FILTER_WINDOW_SIZE; // Return the new average
}


// --- Combined DSP + ML Inference Runner ---
void run_anomaly_inference(float raw_data[3]) 
{
  Serial.printf("\n--- New Reading ---\n");
  
  // 1. Update filter and get smoothed values
  float filtered_V = update_filter(raw_data[0], anom_voltageHistory);
  float filtered_A = update_filter(raw_data[1], anom_currentHistory);
  float filtered_T = update_filter(raw_data[2], anom_tempHistory);
  
  // Advance the shared index for the next run
  anom_filterIndex = (anom_filterIndex + 1) % FILTER_WINDOW_SIZE;

  Serial.printf("Raw Inputs       : [V: %.2f, A: %.2f, C: %.2f]\n", raw_data[0], raw_data[1], raw_data[2]);
  Serial.printf("DSP Filter Output: [V: %.2f, A: %.2f, C: %.2f] (Smoothed)\n", filtered_V, filtered_A, filtered_T);

  // 2. Scale the filtered (smoothed) sensor inputs
  float scaled_input[3];
  scaled_input[0] = (filtered_V - ANOM_VOLTAGE_MIN) / (ANOM_VOLTAGE_MAX - ANOM_VOLTAGE_MIN);
  scaled_input[1] = (filtered_A - ANOM_CURRENT_MIN) / (ANOM_CURRENT_MAX - ANOM_CURRENT_MIN);
  scaled_input[2] = (filtered_T - ANOM_TEMP_MIN) / (ANOM_TEMP_MAX - ANOM_TEMP_MIN);

  // 3. Load into the neural network, run inference, calculate MAE
  anom_input->data.f[0] = scaled_input[0];
  anom_input->data.f[1] = scaled_input[1];
  anom_input->data.f[2] = scaled_input[2];

  unsigned long start_time = micros();

  if (anom_interpreter->Invoke() != kTfLiteOk) 
  { 
    MicroPrintf("Invoke() failed."); 
    xSemaphoreGive(tflmMutex);
    return; 
  }

  unsigned long end_time = micros();
  unsigned long inference_time = end_time - start_time;
  
  Serial.printf("⏱️ Anomaly Inference Time: %lu microseconds\n", inference_time);

  float* reconstructed_output = anom_output->data.f;
  float mae = 0.0;
  for (int i = 0; i < 3; i++) 
  { 
    mae += abs(reconstructed_output[i] - scaled_input[i]); 
  }
  mae /= 3.0;
  Serial.printf("Model MAE Error  : %.4f\n", mae);

  // 4. Make Decision
  if (mae > ANOMALY_THRESHOLD) 
  {
    Serial.println("🚨 -> VERDICT: ANOMALY DETECTED!");
    digitalWrite(PIN_LED, HIGH);
  } 
  else 
  {
    Serial.println("✅ -> VERDICT: System is Normal");
    digitalWrite(PIN_LED, LOW);
  }

}

void simulate_battery_scenario(float raw_data[3]) 
{
  Serial.printf("\n--- New Reading ---\n");
  Serial.printf("Raw Inputs: [V: %.2f, A: %.2f, T: %.2f]\n", raw_data[0], raw_data[1], raw_data[2]);

  // 1. MinMax Scale the inputs to a [0.0, 1.0] range
  float scaled_v = (raw_data[0] - BATT_VOLTAGE_MIN) / (BATT_VOLTAGE_MAX - BATT_VOLTAGE_MIN);
  float scaled_a = (raw_data[1] - BATT_CURRENT_MIN) / (BATT_CURRENT_MAX - BATT_CURRENT_MIN);
  float scaled_t = (raw_data[2] - BATT_TEMP_MIN) / (BATT_TEMP_MAX - BATT_TEMP_MIN);

  // Clamp values to [0.0, 1.0] as a safety measure
  scaled_v = fmax(0.0, fmin(1.0, scaled_v));
  scaled_a = fmax(0.0, fmin(1.0, scaled_a));
  scaled_t = fmax(0.0, fmin(1.0, scaled_t));

  // 2. Load float data directly into the model's input tensor.
  //    (This is correct for your specific hybrid model)
  batt_input->data.f[0] = scaled_v;
  batt_input->data.f[1] = scaled_a;
  batt_input->data.f[2] = scaled_t;

  unsigned long start_time = micros();
  // 3. Run Inference
  if (batt_interpreter->Invoke() != kTfLiteOk) { 
    Serial.println("Invoke() failed."); 
    return; 
  }
  
  unsigned long end_time = micros();
  unsigned long inference_time = end_time - start_time;
  
  Serial.printf("⏱️ Battery Inference Time: %lu microseconds\n", inference_time);

  // 4. Read float data directly from the model's output tensor.
  float soc = batt_output->data.f[0];
  float soh = batt_output->data.f[1];
  
  float final_soc = fmax(0.0, fmin(100.0, soc));
  float final_soh = fmax(0.0, fmin(100.0, soh));
  Serial.printf("🔋 -> VERDICT: State of Charge (SoC): %.2f%% | State of Health (SoH): %.2f%%\n", final_soc, final_soh);
}

void loop() {
  // Empty. FreeRTOS handles execution now.
}
