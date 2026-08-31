# Authorization policy for MCP capability execution.
#
# The platform asks this policy immediately before every capability call. The default is
# deny, so a rule that fails to load or an input the policy does not understand refuses
# the call rather than admitting it.
package research.authz

import rego.v1

# The version reported back in every audit record, so a decision can be traced to the
# policy that made it.
version := "opa-2026-08-01"

default allow := false

allow if {
	count(deny) == 0
}

decision := {
	"allow": allow,
	"reason": reason,
	"policy_version": version,
}

reason := "permitted by policy" if {
	allow
}

reason := concat("; ", sort(deny)) if {
	not allow
}

# --- Refusals -----------------------------------------------------------------

deny contains msg if {
	not input.capability
	msg := "the request does not name a capability"
}

deny contains msg if {
	not has_tenant
	msg := "the request does not name a tenant"
}

deny contains msg if {
	not tenant_permitted
	msg := sprintf("capability is not offered to tenant %v", [input.principal.tenant_id])
}

deny contains msg if {
	not clearance_sufficient
	msg := sprintf(
		"clearance %v cannot reach %v data",
		[clearance, input.capability.max_access_class],
	)
}

deny contains msg if {
	is_agent
	not input.principal.agent_role in input.capability.allowed_agents
	msg := sprintf("agent role %v may not execute this capability", [input.principal.agent_role])
}

deny contains msg if {
	not is_agent
	count(granted_roles) == 0
	msg := "no held role permits this capability"
}

# A side-effecting capability is refused for any agent, whatever the catalogue says.
# Only a human role may cause an external side effect.
deny contains msg if {
	is_agent
	input.capability.side_effecting
	msg := "an agent may not execute a side-effecting capability"
}

# --- Supporting rules ---------------------------------------------------------

has_tenant if {
	input.principal.tenant_id != null
	input.principal.tenant_id != ""
}

is_agent if {
	input.principal.agent_role != null
}

tenant_permitted if {
	"*" in input.capability.tenant_scope
}

tenant_permitted if {
	input.principal.tenant_id in input.capability.tenant_scope
}

clearance := input.principal.clearance

rank := {"public": 0, "internal": 1, "restricted": 2}

clearance_sufficient if {
	rank[clearance] >= rank[input.capability.max_access_class]
}

granted_roles contains role if {
	some role in input.principal.roles
	role in input.capability.required_roles
}
