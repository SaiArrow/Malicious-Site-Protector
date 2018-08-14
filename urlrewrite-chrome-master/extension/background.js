(function() {
	'use strict';

	chrome.webNavigation.onBeforeNavigate.addListener(function(details) {
		chrome.tabs.update(details.tabId, { url: "https://www.google.com" })

	});

})();
