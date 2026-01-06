import json
import regex
import time

from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from openai import OpenAI

from knowledge_graph import Entity, Relationship

MISTRAL_MODEL = 'mistral-large-latest'
OPENAI_MODEL = 'gpt-4o-mini'

REQUEST_PERIOD_MISTRAL = 3

class Model:
	def __init__(self, model='mistral'):
		with open('keys.json') as f:
			keys = json.load(f)
			api_key = keys[model]

		self.last_request = None
		self.client_embeddings = Mistral(api_key=api_key)

		if model == 'mistral':
			self.model_class = MISTRAL_MODEL
			self.client = Mistral(api_key=api_key)
			self.request_period = REQUEST_PERIOD_MISTRAL
		else:
			self.model_class = OPENAI_MODEL
			self.client = OpenAI(api_key=api_key)
			self.request_period = 0

	def __vectorize__(self, input_seq: str):
		success = False

		while not success:
			if self.last_request is not None:
				elapsed = time.time() - self.last_request
				if elapsed < self.request_period:
					time.sleep(self.request_period - elapsed)

			try:
				response = self.client_embeddings.embeddings.create(
					model='mistral-embed', inputs=[input_seq]
				)
				self.last_request = time.time()
				success = True
			except SDKError as err:
				if err.status_code == 429:
					success = False

		return response.data[0].embedding

	def __complete__(self, system: str, user: str):
		if 'mistral' in self.model_class:
			return self.client.chat.complete(
				model=self.model_class,
				messages=[
					{'role': 'system', 'content': system},
					{'role': 'user', 'content': user},
				],
				temperature=0,
			)

		else:
			return self.client.chat.completions.create(
				model=self.model_class,
				messages=[
					{'role': 'system', 'content': system},
					{'role': 'user', 'content': user},
				],
				temperature=0,
			)

	def prompt(self, system: str, user: str) -> str:
		success = False

		while not success:
			if self.last_request is not None:
				elapsed = time.time() - self.last_request
				if elapsed < self.request_period:
					time.sleep(self.request_period - elapsed)

			try:
				response = self.__complete__(system, user)
				self.last_request = time.time()
				success = True
			except SDKError as err:
				if err.status_code == 429:
					success = False

		return response.choices[0].message.content

	### STEP 1 - INFORMATION EXTRACTION ###

	def extract_entities(self, context: str) -> set[Entity]:
		"""
		Takes the context as an input and returns a set of entities parsed from the model's response.
		The result is a set of Entity objects.
		"""

		with open('./prompts/s1/extract_entities_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s1/extract_entities_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(context=context)

		output = self.prompt(system_prompt, user_prompt)
		entities_dict = parse_entities(output)
		return set([Entity(**entity_raw) for entity_raw in entities_dict])

	def extract_relationships(
		self, context: str, entities: list[Entity]
	) -> list[Relationship]:
		"""
		Takes the context and entity list as an input and returns a list of explicit relationships parsed from the model's response.
		The result is a list of Relationship objects containing both top-level and nested triplets.
		"""

		with open('./prompts/s1/extract_explicit_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s1/extract_explicit_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context, entities=str(entities).replace("'", '')
			)

		output = self.prompt(system_prompt, user_prompt)
		relationships_dict = parse_triplets(
			output, parse_recursive=True, return_list=True, parse_quote='grounding'
		)
		return [Relationship(**triplet) for triplet in relationships_dict]

	def infer_implicit(self, context: str, entities: list[Entity]) -> list[Relationship]:
		"""
		Takes the context and entity list as an input and returns a list of implicit relationships parsed from the model's response.
		The result is a list of Relationship objects only containing top-level triplets.
		"""

		with open('./prompts/s1/infer_implicit_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s1/infer_implicit_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context, entities=str(entities).replace("'", '')
			)

		output = self.prompt(system_prompt, user_prompt)
		relationships_dict = parse_triplets(output, inferred=True, return_list=True)
		return [
			Relationship(**triplet)
			for triplet in relationships_dict
			if triplet['id'][0] == 't'
		]

	### STEP 2 - INFERENCE VALIDATION ###

	def challenge_inference(
		self, context: str, inference: Relationship, verbose: bool = False
	) -> tuple[bool, str]:
		"""
		Takes the context and an inferred triplet as an input. Returns a boolean value specifying whether the model
		considers the inference to be reasonable and, if `False`, a brief explanation of the reason for discarding it.
		"""

		with open('./prompts/s2/challenge_inference_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s2/challenge_inference_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context, inference=str(inference).replace("'", '')
			)

		output = self.prompt(system_prompt, user_prompt)
		output_tokenized = regex.findall(
			'(\\byes\\b)|(\\bno\\b)[,;] ([\\w \'".,]+)', output.lower()
		)

		if len(output_tokenized) > 0:
			output_tokenized = output_tokenized[0]
		else:
			print('WARNING: output not parsed correctly for inference challenge')
			print(f'Triplet: {inference}, output: {output}')
			return (True, '')

		if len(output_tokenized[0]) > 0:
			if verbose:
				print(inference, '-> Yes')
			return (True, '')

		explanation = output_tokenized[2].strip()
		if verbose:
			print(inference, '-> No: ', explanation)
		return (False, explanation)

	def correct_inference(
		self, context: str, inference: Relationship, explanation: str, verbose: bool = False
	) -> Relationship | None:
		"""
		Takes the context, an inferred triplet deemed unreasonable and the explanation of the reason it was discarded.
		Returns the Relationship object representing the corrected triplet parsed from the model's response, or `None` if
		the model cannot correct the specified triplet.
		"""

		with open('./prompts/s2/correct_inference_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s2/correct_inference_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context, inference=inference, explanation=explanation
			)

		output = self.prompt(system_prompt, user_prompt)
		res = parse_triplets(output, inferred=True, return_list=True)

		# The parsed result is either one correctly parsed triplet or an empty list if no correction was issued
		if len(res) == 0:
			if verbose:
				print(inference, '~> <none>')
			return None

		if verbose:
			print(inference, '~>', res[0]['rep'])
		return Relationship(**res[0])

	def detect_duplicate(
		self,
		context: str,
		relationships: list[Relationship],
		inference: Relationship,
		verbose: bool = False,
	) -> bool:
		"""
		Takes the context, the list of explicit relationships and an inferred triplet. Returns `True` if the inferred
		triplet is (semantically) a duplicate of one of the explicit relationships, `False` otherwise.
		"""

		relationships_toplevel = [r for r in relationships if not r.is_nested]

		with open('./prompts/s2/detect_duplicate_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s2/detect_duplicate_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context,
				relationships=str(relationships_toplevel).replace("'", ''),
				inference=inference,
			)

		output = self.prompt(system_prompt, user_prompt)
		if output.lower() == 'yes':
			if verbose:
				print(inference, 'removed as duplicate')
			return True

		return False

	def repair_inferences(self, context: str, inferences: list[Relationship]) -> list[Relationship]:
		"""
		Takes the context and the list of Relationship objects representing the top-level inferred triplets.
		Returns a list of Relationship objects representing the repaired triplets parsed from the model's response, where
		duplicates have been removed and badly formed triplets have been fixed. This list includes potential nested triplets.
		"""

		inferences_plain = []
		inferences_nested = []
		for inf in inferences:
			if inf.has_nesting or inf.is_nested:
				inferences_nested.append(inf)
			else:
				inferences_plain.append(inf)

		with open('./prompts/s2/repair_inferences_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s2/repair_inferences_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context, inferences=str(inferences_plain).replace("'", '')
			)

		output = self.prompt(system_prompt, user_prompt)
		parsed_plain = parse_triplets(output, inferred=True, return_list=True)
		parsed_nested = parse_triplets(
			str(inferences_nested).replace("'", ''),
			inferred=True,
			parse_recursive=True,
			return_list=True,
			id_offset=len(parsed_plain),
		)
		res = [Relationship(**t) for t in parsed_plain + parsed_nested]

		return res

	def explain_inference(
		self,
		context: str,
		inference: Relationship,
		relationships: list[Relationship],
		verbose: bool = False,
	) -> list[Relationship]:
		"""
		Takes the context and the Relationship objects representing a top-level inferred triplet and the list of explicit relationships.
		Returns a list of Relationship objects representing explicit relationships that can be interpreted as the premises
		for the inference, parsed from the model's response.
		"""

		with open('./prompts/s2/explain_inference_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s2/explain_inference_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context,
				inference=str(inference).replace("'", ''),
				relationships=str(relationships).replace("'", ''),
			)

		output = self.prompt(system_prompt, user_prompt)
		parsed = [t for t in parse_triplets(output)]

		if verbose:
			premise = '∅' if len(parsed) == 0 else ' && '.join(parsed)
			print(premise, '=>', inference)

		res = [r for r in relationships if r.__repr__() in parsed]
		return res

	### STEP 3 - TIME RELATION EXTRACTION ###

	def classify_events(
		self, context: str, relationships: list[Relationship]
	) -> tuple[list[Relationship], dict]:
		with open('./prompts/s3/classify_event_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s3/classify_event_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context, relationships=str(relationships).replace("'", '')
			)

		output = self.prompt(system_prompt, user_prompt)
		output_parsed = parse_triplets(
			output, parse_tag='event_type', parse_quote='time_reference'
		)

		res = relationships.copy()
		for rel in res:
			if rel.__repr__() in output_parsed:
				out = output_parsed[rel.__repr__()]
				rel.set_event(out[3] != '<state>')
				if out[4].lower() != 'none':
					rel.set_time_reference(out[4])

				output_parsed.pop(rel.__repr__())
			else:
				print(f'WARNING: relationship {rel} not found in keyset {output_parsed.keys()}')

		return res, output_parsed

	def order_temporal(self, context: str, relationships: list[Relationship]) -> dict:
		pairs = [(str(r), str(s)) for r in relationships for s in relationships if r != s]

		with open('./prompts/s3/order_temporal_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s3/order_temporal_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context, pairs=str(pairs).replace("'", '')
			)

		output = self.prompt(system_prompt, user_prompt)
		output_parsed = parse_triplet_pairs(output)
		res = {}
		for r, s, tag in output_parsed:
			if (s, r) in res:
				old_tag = res[(s, r)]
				if (
					old_tag == '<after>'
					and tag != '<before>'
					or old_tag == '<before>'
					and tag != '<after>'
					or old_tag == '<while>'
					and tag != '<while>'
				):
					res[(s, r)] = '<none>'
			else:
				res[(r, s)] = tag

		return res

	### STEP 4 - SCHEMA CANONICALIZATION ###

	def define(self, context: str, triplet: Relationship) -> tuple[str, list[float]]:
		with open('./prompts/s4/define_relationship_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s4/define_relationship_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context,
				triplet=str(triplet).replace("'", ''),
				relationship=triplet.name,
			)

		output = self.prompt(system_prompt, user_prompt)
		return output, self.__vectorize__(output)

	def replace(self, context: str, rel: Relationship, definition: str, candidates: dict) -> str:
		candidates_list = [
			f'{i+1}. {r}: {d} -> {', '.join(rel.rep.split(', ')[:1] + [r] + rel.rep.split(', ')[2:])}'
			for i, (r, d) in enumerate(candidates.items())
		]

		with open('./prompts/s4/replace_relationship_sys.txt', 'r') as sf:
			system_prompt = sf.read()

		with open('./prompts/s4/replace_relationship_usr.txt', 'r') as uf:
			user_prompt = f'{uf.read()}'.format(
				context=context,
				triplet=str(rel).replace("'", ''),
				rel=rel.name,
				definition=definition,
				candidates='\n'.join(candidates_list)
			)

		output = self.prompt(system_prompt, user_prompt)
		return regex.findall('\\w*', output)[0]


def parse_entities(response: str) -> list:
	matches = regex.findall("(\\w[\\w\\s']*) (<[\\w]{3}>)", response)
	return [{'name': m[0], 'type': m[1], 'source': 'text'} for m in matches]


def parse_triplets(
	response: str,
	inferred=False,
	return_list=False,
	parse_recursive=False,
	parse_tag='',
	parse_quote='',
	id_offset=0,
) -> dict | list:
	"""
	This function takes as an input the model's response (a list of possibly nested relational triplets) and parses it
	according to the specified parameters.

	If `return_list` is False, the result will be the a dictionary indexed by the string representation of the triplets
	and whose values are tuples that take the form `(subject, relationship, object[, tag][, quote])`.

	If `return_list` is True, the result will be a list of dictionaries, each of which has the following keys:
	- `id` — a unique identifier, where the first character indicates if the triplet is top-level or nested ('t'/'n') and the second whether it is explicit or inferred ('e'/'i').
	- `subject` — the subject of the triplet.
	- `relationship` — the relationship of the triplet.
	- `object` — the object of the triplet; may be `None` if the triplet represents a unary relation, or the `id` of a nested triplet if applicable.
	- `rep` — the string representation of the whole triplet.
	- `is_inferred` — a flag indicating whether the triplet was inferred.
	- `is_nested` — a flag indicating whether the triplet is nested into another triplet.
	- `has_nesting` — a flag indicating whether the triplet contains a reference to a nested triplet as the object.
	- `[tag]` (optional) — the required tag, if `parse_tag` is non-empty.
	- `[quote]` (optional) — the required quote, if `parse_quote` is non-empty.

	Parameters:
	- `response`: the model's response to be parsed.
	- `inferred`: `True` iff the response contains inferred relationships, as opposed to explicit ones.
	- `return_list`: `True` if the returned value should be a list of dictionaries, `False` if it should be a ditionary.
	- `parse_recursive`: `True` iff nested triplets used as objects should be parsed as standalone triplets (only applies if `return_list` is `True`).
	- `parse_tag`: the name of the field to be parsed as a tag, if any.
	- `parse_quote`: the name of the field to be parsed as a quote, if any.
	"""

	pattern = "(?<triplet>\\(([\\w\\s']+), ([\\w\\s']+), ([\\w\\s'<>]+|(?&triplet))\\))"
	if parse_tag != '':
		pattern += ' (<[\\w]+>)'
	if parse_quote != '':
		pattern += " `([\\w\\s.,:;!?\\-']+)`"

	matches = regex.findall(pattern, response)
	if not return_list:
		return {m[0]: m[1:] for m in matches}

	prefix = 'ti' if inferred else 'te'
	nested_prefix = 'ni' if inferred else 'ne'
	matches = [(prefix + str(i + 1 + id_offset), rel) for i, rel in enumerate(matches)]

	res = []
	counter = 0

	while len(matches) > 0:
		curr_key, curr_rel = matches.pop(0)
		curr_nested = curr_key[0] == 'n'
		rep, sub, rel, obj = curr_rel[:4]
		parsed_object = parse_triplets(obj)

		curr_dict = {
			'id': curr_key,
			'subject': sub,
			'relationship': rel,
			'object': '',
			'rep': rep,
			'is_inferred': inferred,
			'is_nested': curr_nested,
			'has_nesting': False,
		}

		if not curr_nested and parse_tag != '':
			curr_dict[parse_tag] = curr_rel[4]
		if not curr_nested and parse_quote != '':
			curr_dict[parse_quote] = curr_rel[5 if parse_tag != '' else 4]

		if len(parsed_object) > 0:
			curr_dict['has_nesting'] = True

			# Object can be parsed as triplet: create new nested triplet and report its key
			if parse_recursive:
				o_rep = next(iter(parsed_object))
				o_sub, o_rel, o_obj = parsed_object[o_rep]
				counter += 1
				new_key = nested_prefix + str(counter)
				matches.append((new_key, (o_rep, o_sub, o_rel, o_obj)))
				curr_dict['object'] = f'<ref {new_key}>'

			# If nested triplets should not be parsed, copy the string representation only
			else:
				curr_dict['object'] = str(obj)
		else:
			# Object cannot be parsed as triplet: report as is
			curr_dict['object'] = None if obj == '<none>' else obj

		res.append(curr_dict)

	return res


def parse_triplet_pairs(response: str) -> list:
	triplet_pattern = (
		"(?<triplet>\\([\\w\\s']+, [\\w\\s']+, (?:[\\w\\s'<>]+|(?&triplet))\\))"
	)
	pattern = f'\\({triplet_pattern}, ((?&triplet))\\) -> (<(?:before|after|while|none)>)'
	return regex.findall(pattern, response)
