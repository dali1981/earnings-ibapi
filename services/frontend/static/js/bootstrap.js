/**
 * Frontend bootstrapper: autowires DOM elements to controller actions.
 *
 * How it works
 * - Scans the document for any element with a `data-action` attribute.
 * - Looks up a matching function in `window.TradingControllers`.
 * - Binds the proper DOM event:
 *   - `submit` for `<form>` elements (with `preventDefault()`)
 *   - `click` for anything else (buttons, links, etc.)
 * - Wraps controller invocation in a Promise with basic error reporting
 *   (logs to console and shows a simple alert).
 *
 * Requirements
 * - `window.TradingControllers` must be defined (see controllers.js).
 * - `window.TradingAPI` should be present if controllers depend on it.
 *
 * Example
 *   <form data-action="calendar.get"> ... </form>
 *   // controllers.js must export: window.TradingControllers['calendar.get'] = () => {...}
 */

(function () {
  /**
   * Binds a specific DOM element to a controller action.
   *
   * Behavior
   * - Uses `submit` for forms to avoid full page reloads.
   * - Uses `click` for non-forms.
   * - Normalizes controller return to a Promise and catches errors.
   *
   * @param {HTMLElement} el - Target element with data-action
   * @param {string} action - Controller action key (e.g. 'calendar.get')
   */
  function bindAction(el, action) {
    const handler = window.TradingControllers && window.TradingControllers[action];
    if (!handler) return;
    const evt = el.tagName === 'FORM' ? 'submit' : 'click';
    el.addEventListener(evt, (e) => {
      if (evt === 'submit') e.preventDefault();
      Promise.resolve(handler(e)).catch((err) => {
        console.error('Action failed:', action, err);
        alert(err && err.message ? err.message : String(err));
      });
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-action]').forEach((el) => {
      const action = el.getAttribute('data-action');
      if (action) bindAction(el, action);
    });
  });
})();
