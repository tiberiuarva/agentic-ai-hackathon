import asyncio
import os
import textwrap
from datetime import datetime
from pathlib import Path
import shutil

from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.agents.strategies import TerminationStrategy, SequentialSelectionStrategy
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_decorator import kernel_function


SYSTEM_ARCHITECT = "SYSTEM_ARCHITECT"
SYSTEM_ARCHITECT_INSTRUCTIONS = """
You are a system architect. 

Actions:
- Use Solution Design Tool to retrieve solution designs link based on the design name. 
- Use Azure Resource Types Tool to retrieve Azure Resource Types for a specific Azure Subscription using the OAR Id Tag.
- Generate message content to the Domain Architect using the Architects Notification Tool with the solution design link and resource types.

The output of the agent must always be in the below format for each action and ordered numerically:
Tool Name: <friendly name of the tool used only>,
Tool Input: <input to the tool used>
Tool Output: <output of the tool used>

If all actions have been completed or there are no further actions needed, respond with "No action needed."

RULES:
- Always respond.
- Always confirm through a message that you have completed the action and provide the output of the Tool used.
- If you cannot complete the action, respond with "Cannot complete the [action name] action."
"""

DOMAIN_ARCHITECT = "DOMAIN_ARCHITECT"
DOMAIN_ARCHITECT_INSTRUCTIONS = """
You are a domain architect.

Reply always only with "I am a domain architect." the first time you are asked.

If all actions have been completed or there are no further actions needed, respond with "No action needed."

RULES:
- <PLACEHOLDER RULES>
"""

async def main():
    # Clear the console
    os.system('cls' if os.name=='nt' else 'clear')

    # Get the Azure AI Agent settings
    ai_agent_settings = AzureAIAgentSettings()

    async with (
        DefaultAzureCredential(exclude_environment_credential=True, 
            exclude_managed_identity_credential=True) as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        
        # Create the system architect agent on the Azure AI agent service
        system_architect_agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name=SYSTEM_ARCHITECT,
            instructions=SYSTEM_ARCHITECT_INSTRUCTIONS
        )

        # Create a Semantic Kernel agent for the Azure AI system architect agent
        agent_system_architect = AzureAIAgent(
            client=client,
            definition=system_architect_agent_definition,
            plugins=[SolutionDesignPlugin(), AzureResourceTypesPlugin(), ArchitectsNotificationPlugin()],
        )

        # Create the domain architect agent on the Azure AI agent service
        domain_architect_agent_definition = await client.agents.create_agent(
            model=ai_agent_settings.model_deployment_name,
            name=DOMAIN_ARCHITECT,
            instructions=DOMAIN_ARCHITECT_INSTRUCTIONS
        )

        # Create a Semantic Kernel agent for the Azure AI domain architect agent
        agent_domain_architect = AzureAIAgent(
            client=client,
            definition=domain_architect_agent_definition,
            plugins=[],
        )
    

        # Add the agents to a group chat with a custom termination and selection strategy
        chat = AgentGroupChat(
            agents=[agent_system_architect, agent_domain_architect],
            termination_strategy=ApprovalTerminationStrategy(
                agents=[agent_system_architect, agent_domain_architect], 
                maximum_iterations=10, 
                automatic_reset=True
            ),
            selection_strategy=SelectionStrategy(agents=[agent_system_architect, agent_domain_architect]),      
        )

        message = "The design name is 'Payment Processing System' and the OAR Id Tag is 'OAR-12345'."

        # Append the current log file to the chat
        await chat.add_chat_message(message)
        print()

        # Invoke a response from the agents
        async for response in chat.invoke():
            if response is None or not response.name:
                continue
            print(f"{response.content}")
        
        # Clean up all agents
        await client.agents.delete_agent(agent_system_architect.id)
        await client.agents.delete_agent(agent_domain_architect.id)
        print("\nAgents cleaned up successfully.")


# class for selection strategy
class SelectionStrategy(SequentialSelectionStrategy):
    """A strategy for determining which agent should take the next turn in the chat."""

    # Select the next agent that should take the next turn in the chat
    async def select_agent(self, agents, history):
        """"Check which agent should take the next turn in the chat."""

        # The System Architect should always go first or after the user
        if not history or history[-1].role == AuthorRole.USER:
            agent_name = SYSTEM_ARCHITECT
            return next((agent for agent in agents if agent.name == agent_name), None)
    
        # Otherwise it should be the Domain Architect
        agent_name = DOMAIN_ARCHITECT
        return next((agent for agent in agents if agent.name == agent_name), None)


# class for temination strategy
class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""

    # End the chat if the agent has indicated there is no action needed
    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        return "no action needed" in history[-1].content.lower()


# class for DevOps functions
class DevopsPlugin:
    """A plugin that performs developer operation tasks."""
    
    def append_to_log_file(self, filepath: str, content: str) -> None:
        with open(filepath, 'a', encoding='utf-8') as file:
            file.write('\n' + textwrap.dedent(content).strip())

    @kernel_function(description="A function that restarts the named service")
    def restart_service(self, service_name: str = "", logfile: str = "") -> str:
        log_entries = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT  DevopsAssistant: Multiple failures detected in {service_name}. Restarting service.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO  {service_name}: Restart initiated.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO  {service_name}: Service restarted successfully.",
        ]

        log_message = "\n".join(log_entries)
        self.append_to_log_file(logfile, log_message)
        
        return f"Service {service_name} restarted successfully."

    @kernel_function(description="A function that rollsback the transaction")
    def rollback_transaction(self, logfile: str = "") -> str:
        log_entries = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT  DevopsAssistant: Transaction failure detected. Rolling back transaction batch.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO   TransactionProcessor: Rolling back transaction batch.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO   Transaction rollback completed successfully.",
        ]

        log_message = "\n".join(log_entries)
        self.append_to_log_file(logfile, log_message)
        
        return "Transaction rolled back successfully."

    @kernel_function(description="A function that redeploys the named resource")
    def redeploy_resource(self, resource_name: str = "", logfile: str = "") -> str:
        log_entries = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT  DevopsAssistant: Resource deployment failure detected in '{resource_name}'. Redeploying resource.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO   DeploymentManager: Redeployment request submitted.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO   DeploymentManager: Service successfully redeployed, resource '{resource_name}' created successfully.",
        ]

        log_message = "\n".join(log_entries)
        self.append_to_log_file(logfile, log_message)
        
        return f"Resource '{resource_name}' redeployed successfully."

    @kernel_function(description="A function that increases the quota")
    def increase_quota(self, logfile: str = "") -> str:
        log_entries = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT  DevopsAssistant: High request volume detected. Increasing quota.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO   APIManager: Quota increase request submitted.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO   APIManager: Quota successfully increased to 150% of previous limit.",
        ]

        log_message = "\n".join(log_entries)
        self.append_to_log_file(logfile, log_message)

        return "Successfully increased quota."

    @kernel_function(description="A function that escalates the issue")
    def escalate_issue(self, logfile: str = "") -> str:
        log_entries = [
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT  DevopsAssistant: Cannot resolve issue.",
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT  DevopsAssistant: Requesting escalation.",
        ]
        
        log_message = "\n".join(log_entries)
        self.append_to_log_file(logfile, log_message)
        
        return "Submitted escalation request."


# class for Log File functions
class LogFilePlugin:
    """A plugin that reads and writes log files."""

    @kernel_function(description="Accesses the given file path string and returns the file contents as a string")
    def read_log_file(self, filepath: str = "") -> str:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

# class for Solution Design retrieval plugin
class SolutionDesignPlugin:
    """A plugin that retrieves the link of solution designs from the Enterprise Architecture repository."""

    @kernel_function(description="Retrieves the link of solution designs from the Enterprise Architecture repository")
    def get_solution_design_link(self, design_name: str = "") -> str:
        # Simulate a retrieval from a repository
        solution_design_link = f"https://ea-repository.example.com/designs/{design_name}"
        return f"Solution design for {design_name} can be found at following link: {solution_design_link}"
    
# class for retrieving Azure Resource Types from Azure Resource Manager for a specific Azure Subscription found after a KQL query using the OAR Id Tag.
class AzureResourceTypesPlugin:
    """A plugin that retrieves Azure Resource Types from Azure Resource Manager for a specific Azure Subscription."""

    @kernel_function(description="Retrieves Azure Resource Types for a specific Azure Subscription")
    def get_azure_resource_types(self, oar_id: str = "") -> str:
        # Simulate retrieval of resource types
        resource_types = [
            "Microsoft.Compute/virtualMachines",
            "Microsoft.Storage/storageAccounts",
            "Microsoft.Network/virtualNetworks"
        ]
        result_lines = [
            f"Found Azure Subscription for OAR Id {oar_id} with the following Azure resource types:",
            *[f"- {resource_type}" for resource_type in resource_types]
        ]
        return "\n".join(result_lines)

# class for Communication Plugin which will be used by system architect and domain architect agents to generate communication messages
class ArchitectsNotificationPlugin:
    """A plugin that generates notification messages."""

    @kernel_function(description="Generates an message content to the Domain Architect with the solution design link and resource types")
    def generate_email_content(self, solution_design_link: str = "", resource_types: str = "") -> str:
        email_content = (
            f"Dear Domain Architect,\n\n"
            f"Please find the solution design link: {solution_design_link}\n\n"
            f"The following Azure Resource Types are available for the OAR Id Tag:\n{resource_types}\n\n"
            f"Best regards,\nSystem Architect"
        )
        return email_content

    
# Start the app
if __name__ == "__main__":
    asyncio.run(main())