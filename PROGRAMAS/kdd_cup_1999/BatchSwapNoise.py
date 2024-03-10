import numpy as np


def batch_swap_noise(x, swap_percentage):
    """
    Aplica BatchSwapNoise a los datos de entrada x.

    Parameters:
        x (numpy.ndarray): Los datos de entrada.
        swap_percentage (float): El porcentaje de características a intercambiar en cada fila.

    Returns:
        numpy.ndarray: Los datos de entrada con BatchSwapNoise aplicado.
    """
    num_samples, num_features = x.shape
    num_swaps = int(swap_percentage * num_features)

    corrupted_x = x.copy()

    for i in range(num_samples):
        # Seleccionar las características a intercambiar
        swap_indices = np.random.choice(num_features, size=num_swaps, replace=False)

        # Seleccionar otra fila aleatoria para intercambiar las características
        other_sample_index = np.random.randint(num_samples)

        # Intercambiar las características seleccionadas con la otra fila
        corrupted_x[i, swap_indices] = x[other_sample_index, swap_indices]

    return corrupted_x


# Ejemplo de uso:
# Definir datos de entrada x
x = np.array([[1, 2, 3, 4,5,6,7,8,9,10,11,12,13],
              [5, 6, 7, 8,9,10,11,12,13,14,15,16,17],
              [9, 10, 11, 12,13,14,15,16,17,18,19,20,21]])

# Aplicar BatchSwapNoise con un 15% de intercambio
swap_percentage = 0.15
corrupted_x = batch_swap_noise(x, swap_percentage)
print("Datos de entrada originales:")
print(x)
print("\nDatos de entrada con BatchSwapNoise aplicado:")
print(corrupted_x)


# Lista de elementos proporcionada en el texto
elementos = [
    "http", "smtp", "finger", "domain_u", "auth", "telnet", "ftp", "eco_i", "ntp_u", "ecr_i",
    "other", "private", "pop_3", "ftp_data", "rje", "time", "mtp", "link", "remote_job", "gopher",
    "ssh", "name", "whois", "domain", "login", "imap4", "daytime", "ctf", "nntp", "shell", "IRC",
    "nnsp", "http_443", "exec", "printer", "efs", "courier", "uucp", "klogin", "kshell", "echo",
    "discard", "systat", "supdup", "iso_tsap", "hostnames", "csnet_ns", "pop_2", "sunrpc", "uucp_path",
    "netbios_ns", "netbios_ssn", "netbios_dgm", "sql_net", "vmnet", "bgp", "Z39_50", "ldap", "netstat",
    "urh_i", "X11", "urp_i", "pm_dump", "tftp_u", "tim_i", "red_i"
]

# Contar los elementos
total_elementos = len(elementos)

print("Total de elementos:", total_elementos)
