#include "avdweb_AnalogReadFast.h"

#define EMG_CHANNEL_0 A0
#define EMG_CHANNEL_1 A1
#define EMG_CHANNEL_2 A2
#define SAMPLING_PERIOD 1250
#define BAUD_RATE 230400

unsigned long past = 0;

void setup(void)
{
  Serial.begin(BAUD_RATE);

  #if defined(__arm__)
  analogReadResolution(12);  
  #endif

  analogRead(EMG_CHANNEL_0);
  analogRead(EMG_CHANNEL_1);
  analogRead(EMG_CHANNEL_2);
}

void loop(void)
{
  unsigned long present = micros();

  if (present - past >= SAMPLING_PERIOD) 
  {
    past = present;

    float adc_ch_0 = analogReadFast(EMG_CHANNEL_0);
    float adc_ch_1 = analogReadFast(EMG_CHANNEL_1);
    float adc_ch_2 = analogReadFast(EMG_CHANNEL_2);

    float adc_ch_0_mV = adc_ch_0*(3.3/4095.0)*1000.0;
    float adc_ch_1_mV = adc_ch_1*(3.3/4095.0)*1000.0;
    float adc_ch_2_mV = adc_ch_2*(3.3/4095.0)*1000.0;


    Serial.print(adc_ch_0_mV);
    Serial.print(",");
    Serial.print(adc_ch_1_mV);
    Serial.print(",");
    Serial.println(adc_ch_2_mV);
  }
}


