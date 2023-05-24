import requests
import json

url = "https://api.innov.vps.energy/api/1.4/sessions"

payload = {
    "Login": "cwll_isep",
    "Password": "LivingLabs2023"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

import xml.etree.ElementTree as ET

# The given XML string
xml_string = response.text

# Parse the XML string into an ElementTree object
root = ET.fromstring(xml_string)

print(xml_string)

# Get the values of each child element
needs_user_consent = root.findtext('.//{http://schemas.datacontract.org/2004/07/VPS.iEnergy.Entities.Sessions}NeedsUserConsent')
refresh_timeout = root.findtext('.//{http://schemas.datacontract.org/2004/07/VPS.iEnergy.Entities.Sessions}RefreshTimeout')
refresh_token = root.findtext('.//{http://schemas.datacontract.org/2004/07/VPS.iEnergy.Entities.Sessions}RefreshToken')
timeout = root.findtext('.//{http://schemas.datacontract.org/2004/07/VPS.iEnergy.Entities.Sessions}Timeout')
token = root.findtext('.//{http://schemas.datacontract.org/2004/07/VPS.iEnergy.Entities.Sessions}Token')

# Print the values
print('NeedsUserConsent:', needs_user_consent)
print('RefreshTimeout:', refresh_timeout)
print('RefreshToken:', refresh_token)
print('Timeout:', timeout)
print('Token:', token)
