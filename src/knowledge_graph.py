from enum import Enum
import os

class Entity:
	def __init__(self, **kwargs):
		self.name = kwargs['name']
		self.type = EntityType(kwargs['type'])

	def __repr__(self):
		return self.name

	def __eq__(self, value):
		if type(value) is not Entity:
			return False
		return self.name == value.name

	def __hash__(self):
		return hash(self.name)

	def asjson(self) -> str:
		res = '{'
		res += f'"name":"{self.name}","type":"{self.type}"'
		res += '}'
		return res

class Relationship:
	def __init__(self, **kwargs):
		self.id = kwargs['id']

		self.name = kwargs['relationship']
		self.previous_name = None
		self.subject = kwargs['subject']
		self.object = kwargs['object']
		self.rep = kwargs['rep']

		self.is_inferred = kwargs['is_inferred']
		self.is_nested = kwargs['is_nested']
		self.has_nesting = kwargs['has_nesting']

		self.is_event = None
		self.time_reference = None
		self.text_reference = kwargs['grounding'] if 'grounding' in kwargs else None

		self.is_discarded = False
		self.discard_reason = None
		self.correction = None

	def __repr__(self):
		return self.rep

	def asjson(self) -> str:
		res = '{'
		res += f'"id":"{self.id}","triplet":"{self.rep}","is_inferred":{str(self.is_inferred).lower()}'

		if self.previous_name is not None:
			res += f',"previous_name":"{str(self.previous_name)}"'

		if self.is_event is not None:
			res += f',"is_event":{str(self.is_event).lower()}'

		if self.time_reference is not None:
			res += f',"time_reference":"{self.time_reference}"'
		if self.text_reference is not None:
			res += f',"text_reference":"{self.text_reference}"'

		if self.is_discarded:
			res += f',"discarded": true, "reason": "{self.discard_reason}"'
			if self.correction is not None:
				res += f',"correction": "{self.correction}"'

		res += '}'
		return res

	def update_name(self, name: str):
		self.previous_name = self.name
		self.name = name
		splits = self.rep.split(', ')
		splits[1] = name
		self.rep = ', '.join(splits)

	def set_event(self, is_event: bool):
		if self.is_event is not None:
			error_msg = 'Event flag has already been set and cannot be updated.\n'
			error_msg += f'Triplet {self} represents {"an event" if self.is_event else "a state"}'
			raise ValueError(error_msg)

		self.is_event = is_event

	def set_time_reference(self, time_ref: str):
		if self.time_reference is not None:
			error_msg = 'Time reference has already been set and cannot be updated.\n'
			error_msg += f'Triplet {self} with time reference {self.time_reference}'
			raise ValueError(error_msg)

		self.time_reference = time_ref

	def discard(self, reason: str, correction=None):
		if self.is_discarded:
			raise ValueError(f'Triplet {self} has already been discarded')

		self.is_discarded = True
		self.discard_reason = reason
		self.correction = correction

class TimeReference:
	def __init__(self, is_lasting: bool, duration: str = '', start: str = '', end: str = ''):
		self.is_lasting = is_lasting
		self.duration = duration if duration != '' else None
		self.start = start if start != '' else None
		self.end = end if end != '' else None

class Explanation:
	def __init__(self, premises: list[Relationship], conclusion: Relationship):
		self.premises = premises.copy()
		self.conclusion = conclusion

	def __repr__(self):
		res = ''
		res += ' && '.join(self.premises)
		res += ' => '
		res += str(self.conclusion)
		return res

class TimeRelation:
	def __init__(self, first_event: Relationship, second_event: Relationship, simultaneous: bool):
		self.first_event = first_event
		self.second_event = second_event
		self.simultaneous = simultaneous

	def __repr__(self):
		res = self.first_event
		res += ' while ' if self.simultaneous else ' before '
		res += self.second_event
		return res

class EntityType(Enum):
	PERSON = '<per>'
	ANIMAL = '<ani>'
	ORGANIZATION = '<org>'
	GEO_POLITICAL_ENTITY = '<gpe>'
	FACILITY = '<fac>'
	OBJECT = '<obj>'
	OCCUPATION = '<occ>'
	TIME = '<tim>'
	NUMBER = '<num>'
	MISC = '<msc>'
	IMPLICIT = '<imp>'