#include <stdio.h>
#include <stdbool.h>
#include "xtimer.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netreg.h"
#include "net/gnrc/pktdump.h"
#include "net/netopt.h"

int main(void)
{
    xtimer_sleep(3); // Wait for USB
    puts("S2 RECEIVER: Online.");

    gnrc_netif_t *netif = NULL;

    // Retry loop: Keep looking for the radio for 10 seconds
    for (int i = 0; i < 10; i++) {
        netif = gnrc_netif_iter(NULL);
        if (netif != NULL) break;
        puts("Waiting for radio hardware...");
        xtimer_sleep(1);
    }

    if (netif == NULL) {
        puts("FATAL ERROR: Radio hardware not found after 10s.");
        return 1;
    }

    // Set Promiscuous Mode
    bool enable = true;
    gnrc_netapi_set(netif->pid, NETOPT_PROMISCUOUSMODE, 0, &enable, sizeof(enable));
    
    // Register pktdump
    gnrc_netreg_entry_t dump = GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, gnrc_pktdump_pid);
    gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &dump);

    puts("S2 RECEIVER: Sniffer engaged. Waiting for S1...");

    while (1) {
        xtimer_sleep(10);
    }
    return 0;
}
