$token = "paste_your_token_here"
Invoke-WebRequest -Uri 'https://hcsc-sb.pvcloud.com/odataservice/odataservice.svc/Automart_project_dim?$top=1&$format=json' -Headers @{Authorization="Bearer $token"} -UseBasicParsing
