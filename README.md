# Workflow


## Usando VPN

1. Ejecutar el shell script etc/sdr-vpn/vpn_reconnector.sh, que ejecuta etc/sdr-vpn/sdr_vpn_start.sh cuando detecta flags enviadas por vpn_utils en /tmp/vpn-flags.

2. Generar un objeto VPNKeepAlive del módulo src.core.vpn_utils, para poder:

- Reconectar con "`vpnkeepalive_obj.reconnect()`": Solo al principio Y de manera opcional, si se sospecha que pasó mas de 10 minutos de la ultima reconexión se agrega esta línea. Evitar ejecutarla en bucles, ya que muchos reconnects pueden provocar soft ban por parte del servidor VPN.

- Reconectar "quizás" con "`vpnkeepalive_obj.maybe_reconnect()"`: Solo cuando se cumpla un plazo de tiempo y en un lugar que sea seguro (es decir, que no sea un paso intermedio que depende de un mismo objeto SDR), generalmente entre 30 y 40 minutos. De esta forma, se soluciona el problema del Rekey espurio de openconnect.

- Desconectar con "`vpnkeepalive_obj.disconnect()`": Para no acaparar la conexión VPN, añadir "vpnkeepalive_obj.disconnect()"



