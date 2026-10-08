# The existing I2C driver toggles dri_clk every five 10-MHz source cycles.
# CLK_FREQ=10MHz, I2C_FREQ=250kHz => dri_clk=1MHz, SCL=250kHz.
create_generated_clock -name hdmi_i2c_driver \
    -source [get_pins u_config/u_i2c_dri/dri_clk_reg/C] \
    -divide_by 10 [get_pins u_config/u_i2c_dri/dri_clk_reg/Q]

# ODDRE1 D1=0,D2=1: rising board sample clock is half a pixel cycle later.
create_generated_clock -name hdmi_video_forwarded \
    -source [get_pins u_video_clk/C] -divide_by 1 -invert [get_ports video_clk]

# External pushbutton is asynchronous. Each clock domain synchronizes release.
set_false_path -from [get_ports sys_rst_n]

# Board-level output delays must come from the KU060/MS7210 setup/hold and
# trace-skew budget. Do not invent zeros or false-path the RGB interface.
# A board-specific XDC can supply:
# set_output_delay -clock hdmi_video_forwarded -max <setup+data_max-clock_min> ...
# set_output_delay -clock hdmi_video_forwarded -min <data_min-clock_max-hold> ...
