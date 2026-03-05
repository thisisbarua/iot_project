APPLICATION = iot_receiver
BOARD ?= adafruit-feather-nrf52840-sense
RIOTBASE ?= $(CURDIR)/../RIOT

# Core drivers and stack
USEMODULE += netdev_default
USEMODULE += auto_init_gnrc_netif
USEMODULE += gnrc_ipv6_default
USEMODULE += gnrc_pktdump
USEMODULE += gnrc_netreg
USEMODULE += gnrc_netapi
USEMODULE += gnrc_netif
USEMODULE += xtimer

# Settings
DEFAULT_CHANNEL = 26
PAN_ID = 0xABCD

include $(RIOTBASE)/Makefile.include
