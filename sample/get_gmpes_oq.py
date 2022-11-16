from Hazard.psha import derive_openquake_info

gmpes = derive_openquake_info()
print(gmpes)

for gmpe in gmpes:
    try:
        derive_openquake_info(gmpe)
        print("\n")
    except:
        continue

"""
ML-based GMPEs
    DerrasEtAl2014
    
"""
