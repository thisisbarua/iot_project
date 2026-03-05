#include <string.h>
#include "xtimer.h"
#include "net/sock/udp.h"
#include "net/ipv6/addr.h"
#include "periph/gpio.h"

#define INTERVAL (100000U) // 100ms = 10 packets/second
#define PORT     (8888)

int main(void)
{
    // Wait 2 seconds just for stability
    xtimer_sleep(2); 

    sock_udp_ep_t remote = { .family = AF_INET6, .port = PORT };
    ipv6_addr_set_all_nodes_multicast((ipv6_addr_t *)&remote.addr.ipv6, IPV6_ADDR_MCAST_SCP_LINK_LOCAL);

    char buf[64];
    uint32_t seq = 0;
    sock_udp_t sock;

    // Create socket silently
    sock_udp_create(&sock, NULL, NULL, 0);
    gpio_init(LED0_PIN, GPIO_OUT);

    while (1) {
        // Create the sequence message
        sprintf(buf, "SEQ:%lu", (unsigned long)seq);
        
        // Broadcast blindly (no printfs!)
        sock_udp_send(&sock, buf, strlen(buf), &remote);
        gpio_toggle(LED0_PIN); 

        seq++;
        xtimer_usleep(INTERVAL);
    }

    return 0;
}