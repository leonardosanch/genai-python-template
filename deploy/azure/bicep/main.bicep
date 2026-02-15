// Azure Bicep template for Container Apps deployment

@description('Location for all resources')
param location string = resourceGroup().location

@description('Container Apps Environment name')
param environmentName string = 'agent-env'

@description('Container App name')
param containerAppName string = 'agent-app'

@description('Container image')
param containerImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'

@description('Cosmos DB account name')
param cosmosAccountName string = 'cosmos-${uniqueString(resourceGroup().id)}'

@description('Azure OpenAI endpoint')
param azureOpenAIEndpoint string

@description('Azure OpenAI deployment name')
param azureOpenAIDeployment string = 'gpt-35-turbo'

@description('Service Bus namespace')
param serviceBusNamespace string = 'sb-${uniqueString(resourceGroup().id)}'

// Container Apps Environment
resource environment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'logs-${environmentName}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Cosmos DB Account
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: cosmosAccountName
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
      }
    ]
    capabilities: [
      {
        name: 'EnableServerless'  // Serverless for cost optimization
      }
    ]
  }
}

// Cosmos DB Database
resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-04-15' = {
  parent: cosmosAccount
  name: 'agents'
  properties: {
    resource: {
      id: 'agents'
    }
  }
}

// Cosmos DB Container
resource cosmosContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = {
  parent: cosmosDatabase
  name: 'conversations'
  properties: {
    resource: {
      id: 'conversations'
      partitionKey: {
        paths: [
          '/conversation_id'
        ]
        kind: 'Hash'
      }
      defaultTtl: 2592000  // 30 days
    }
  }
}

// Service Bus Namespace
resource serviceBus 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: serviceBusNamespace
  location: location
  sku: {
    name: 'Standard'
  }
}

// Service Bus Queue
resource serviceBusQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: serviceBus
  name: 'agent-events'
  properties: {
    maxSizeInMegabytes: 1024
    defaultMessageTimeToLive: 'P7D'  // 7 days
  }
}

// Managed Identity for Container App
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${containerAppName}-identity'
  location: location
}

// Role assignment: Cosmos DB Data Contributor
resource cosmosRoleAssignment 'Microsoft.DocumentDB/databaseAccounts/sqlRoleAssignments@2023-04-15' = {
  parent: cosmosAccount
  name: guid(cosmosAccount.id, managedIdentity.id, 'contributor')
  properties: {
    roleDefinitionId: '${cosmosAccount.id}/sqlRoleDefinitions/00000000-0000-0000-0000-000000000002'
    principalId: managedIdentity.properties.principalId
    scope: cosmosAccount.id
  }
}

// Role assignment: Service Bus Data Sender
resource serviceBusRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(serviceBus.id, managedIdentity.id, 'sender')
  scope: serviceBus
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '69a216fc-b8fb-44d8-bc22-1f3c2cd27a39')  // Azure Service Bus Data Sender
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        transport: 'http'
        allowInsecure: false
      }
      secrets: [
        {
          name: 'azure-openai-key'
          value: 'your-azure-openai-key-here'  // Use Key Vault in production
        }
      ]
    }
    template: {
      containers: [
        {
          name: containerAppName
          image: containerImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'COSMOS_ENDPOINT'
              value: cosmosAccount.properties.documentEndpoint
            }
            {
              name: 'COSMOS_DATABASE'
              value: 'agents'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAIEndpoint
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT'
              value: azureOpenAIDeployment
            }
            {
              name: 'AZURE_OPENAI_KEY'
              secretRef: 'azure-openai-key'
            }
            {
              name: 'SERVICEBUS_NAMESPACE'
              value: '${serviceBusNamespace}.servicebus.windows.net'
            }
            {
              name: 'PORT'
              value: '8080'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0  // Scale to zero
        maxReplicas: 10
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

// Outputs
output containerAppFQDN string = containerApp.properties.configuration.ingress.fqdn
output cosmosEndpoint string = cosmosAccount.properties.documentEndpoint
output serviceBusNamespace string = serviceBus.name
output managedIdentityClientId string = managedIdentity.properties.clientId
