package research.authz_test

import data.research.authz
import rego.v1

fetch_capability := {
	"server": "web-research",
	"name": "fetch",
	"max_access_class": "public",
	"allowed_agents": ["researcher"],
	"required_roles": ["requester"],
	"tenant_scope": ["*"],
	"side_effecting": false,
	"requires_approval": false,
}

requester := {
	"tenant_id": "acme",
	"subject_id": "user-1",
	"roles": ["requester"],
	"agent_role": null,
	"clearance": "public",
}

researcher := object.union(requester, {"agent_role": "researcher"})

request(principal, capability) := {"principal": principal, "capability": capability}

test_a_requester_may_fetch if {
	authz.allow with input as request(requester, fetch_capability)
}

test_an_allowed_agent_may_fetch if {
	authz.allow with input as request(researcher, fetch_capability)
}

test_the_decision_reports_the_policy_version if {
	decision := authz.decision with input as request(requester, fetch_capability)
	decision.policy_version == authz.version
	decision.reason == "permitted by policy"
}

test_an_empty_input_is_refused if {
	not authz.allow with input as {}
}

test_a_missing_capability_is_refused if {
	decision := authz.decision with input as {"principal": requester}
	not decision.allow
	contains(decision.reason, "does not name a capability")
}

test_a_missing_tenant_is_refused if {
	principal := object.union(requester, {"tenant_id": ""})
	not authz.allow with input as request(principal, fetch_capability)
}

test_another_tenant_scope_is_refused if {
	capability := object.union(fetch_capability, {"tenant_scope": ["globex"]})
	decision := authz.decision with input as request(requester, capability)
	not decision.allow
	contains(decision.reason, "not offered to tenant acme")
}

test_a_scoped_tenant_is_permitted if {
	capability := object.union(fetch_capability, {"tenant_scope": ["acme"]})
	authz.allow with input as request(requester, capability)
}

test_insufficient_clearance_is_refused if {
	capability := object.union(fetch_capability, {"max_access_class": "restricted"})
	decision := authz.decision with input as request(requester, capability)
	not decision.allow
	contains(decision.reason, "cannot reach restricted data")
}

test_sufficient_clearance_is_permitted if {
	capability := object.union(fetch_capability, {"max_access_class": "internal"})
	principal := object.union(requester, {"clearance": "internal"})
	authz.allow with input as request(principal, capability)
}

test_a_higher_clearance_covers_a_lower_class if {
	principal := object.union(requester, {"clearance": "restricted"})
	authz.allow with input as request(principal, fetch_capability)
}

test_an_unlisted_agent_role_is_refused if {
	principal := object.union(requester, {"agent_role": "reporter"})
	decision := authz.decision with input as request(principal, fetch_capability)
	not decision.allow
	contains(decision.reason, "agent role reporter may not execute")
}

test_the_planner_may_not_execute if {
	principal := object.union(requester, {"agent_role": "planner"})
	not authz.allow with input as request(principal, fetch_capability)
}

test_a_role_the_caller_does_not_hold_is_refused if {
	capability := object.union(fetch_capability, {"required_roles": ["administrator"]})
	decision := authz.decision with input as request(requester, capability)
	not decision.allow
	contains(decision.reason, "no held role permits")
}

test_an_agent_may_never_cause_a_side_effect if {
	capability := object.union(fetch_capability, {
		"side_effecting": true,
		"requires_approval": true,
		"allowed_agents": ["researcher"],
	})
	decision := authz.decision with input as request(researcher, capability)
	not decision.allow
	contains(decision.reason, "may not execute a side-effecting capability")
}

test_a_human_may_cause_an_approved_side_effect if {
	capability := object.union(fetch_capability, {
		"side_effecting": true,
		"requires_approval": true,
		"required_roles": ["administrator"],
		"max_access_class": "restricted",
	})
	principal := object.union(requester, {
		"roles": ["administrator"],
		"clearance": "restricted",
	})
	authz.allow with input as request(principal, capability)
}

test_every_refusal_is_reported if {
	capability := object.union(fetch_capability, {
		"tenant_scope": ["globex"],
		"max_access_class": "restricted",
	})
	decision := authz.decision with input as request(requester, capability)
	not decision.allow
	contains(decision.reason, "not offered to tenant")
	contains(decision.reason, "cannot reach restricted data")
}
