"""Brief Tutor workflow package."""

import truststore

from graph.console_log import configure_stdio

# CrowdStrike Falcon re-signs some HTTPS calls with "Falcon ROOT CA Proxy".
# That root is in the Windows store, not in certifi, so Python clients fail
# verification until they use the OS trust store.
truststore.inject_into_ssl()
configure_stdio()
