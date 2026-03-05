APPLICATION = iot_sender
BOARD ?= adafruit-feather-nrf52840-sense
RIOTBASE ?= $(CURDIR)/../RIOT

# Modules for networking and timing
USEMODULE += netdev_default
USEMODULE += auto_init_gnrc_netif
USEMODULE += gnrc_ipv6_default
USEMODULE += gnrc_udp
USEMODULE += gnrc_sock_udp
USEMODULE += xtimer

# Force specific radio settings
DEFAULT_CHANNEL = 26
PAN_ID = 0xABCD

include $(RIOTBASE)/Makefile.include
